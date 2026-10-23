// Common task-complete CUDA benchmark: Global-UF vs FlatZone-OLE, v0.1
//
// Both algorithms:
//   - receive the same resident or host grayscale image;
//   - produce full 8-connected regional-max and regional-min masks;
//   - include initialization, equal-flat-zone labeling, boundary flag
//     computation, root reduction, and mask materialization;
//   - are cross-checked pixelwise after benchmarking.
//
// Timing modes:
//   compute_ms: preallocated buffers, input already on device, full compute.
//   h2h_ms: reusable allocation, H2D input + full compute + D2H two masks.
//
// The FlatZone-OLE route implements the iterative label-equivalence algorithm of
// Kalentev et al., J. Parallel Distrib. Comput. 71(4):615-620, 2011, following the
// reference implementation distributed with YACCLAB (BSD 3-Clause), including its
// convergence protocol, in which the iteration flag is written and read back
// through the host once per pass. It is adapted here to grayscale equal-valued
// flat zones and to the exact dual-mask output contract.

#define NOMINMAX
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error_code = (call);                                  \
        if (error_code != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(error_code));            \
        }                                                                       \
    } while (0)

struct DeviceBuffers {
    std::uint8_t* image = nullptr;
    int* labels = nullptr;
    std::uint8_t* has_greater = nullptr;
    std::uint8_t* has_smaller = nullptr;
    std::uint32_t* root_any_greater = nullptr;
    std::uint32_t* root_any_smaller = nullptr;
    std::uint8_t* maxima = nullptr;
    std::uint8_t* minima = nullptr;
    int* changed = nullptr;
};

struct TimingSummary {
    double median_compute_ms = 0.0;
    double p95_compute_ms = 0.0;
    double median_h2h_ms = 0.0;
    double p95_h2h_ms = 0.0;
    double median_iterations = 1.0;
    double p95_iterations = 1.0;
};

static bool ends_with_ci(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) {
        return false;
    }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        const char a = static_cast<char>(std::tolower(
            static_cast<unsigned char>(
                value[value.size() - suffix.size() + i])));
        const char b = static_cast<char>(std::tolower(
            static_cast<unsigned char>(suffix[i])));
        if (a != b) {
            return false;
        }
    }
    return true;
}

static std::vector<fs::path> list_images(const fs::path& folder) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string path = entry.path().string();
        if (ends_with_ci(path, ".png") || ends_with_ci(path, ".jpg") ||
            ends_with_ci(path, ".jpeg") || ends_with_ci(path, ".pgm") ||
            ends_with_ci(path, ".bmp") || ends_with_ci(path, ".tif") ||
            ends_with_ci(path, ".tiff")) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

__global__ void common_compute_boundary_flags(
    const std::uint8_t* image,
    std::uint8_t* has_greater,
    std::uint8_t* has_smaller,
    int width,
    int height
) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const int index = y * width + x;
    const std::uint8_t value = image[index];
    std::uint8_t greater = 0;
    std::uint8_t smaller = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= height) {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            const int nx = x + dx;
            if (nx < 0 || nx >= width) {
                continue;
            }
            const std::uint8_t neighbour = image[ny * width + nx];
            greater |= static_cast<std::uint8_t>(neighbour > value);
            smaller |= static_cast<std::uint8_t>(neighbour < value);
        }
    }
    has_greater[index] = greater;
    has_smaller[index] = smaller;
}

__global__ void common_reduce_flags(
    const int* labels_zero_based,
    const std::uint8_t* has_greater,
    const std::uint8_t* has_smaller,
    std::uint32_t* root_any_greater,
    std::uint32_t* root_any_smaller,
    int count
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) {
        return;
    }
    const int root = labels_zero_based[index];
    if (has_greater[index] != 0) {
        atomicOr(root_any_greater + root, 1u);
    }
    if (has_smaller[index] != 0) {
        atomicOr(root_any_smaller + root, 1u);
    }
}

__global__ void common_materialize_masks(
    const int* labels_zero_based,
    const std::uint32_t* root_any_greater,
    const std::uint32_t* root_any_smaller,
    std::uint8_t* maxima,
    std::uint8_t* minima,
    int count
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) {
        return;
    }
    const int root = labels_zero_based[index];
    maxima[index] = root_any_greater[root] == 0 ? 255 : 0;
    minima[index] = root_any_smaller[root] == 0 ? 255 : 0;
}

// ---------------------------------------------------------------------------
// Frozen Global-UF CUDA route
// ---------------------------------------------------------------------------

__global__ void guf_init_parent(int* parent, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        parent[index] = index;
    }
}

__device__ __forceinline__ int guf_find_root(const int* parent, int node) {
    int root = parent[node];
    while (root != parent[root]) {
        root = parent[root];
    }
    return root;
}

__device__ __forceinline__ void guf_union_minroot(
    int* parent,
    int first,
    int second
) {
    while (true) {
        const int root_first = guf_find_root(parent, first);
        const int root_second = guf_find_root(parent, second);
        if (root_first == root_second) {
            return;
        }
        const int high = max(root_first, root_second);
        const int low = min(root_first, root_second);
        const int old = atomicCAS(parent + high, high, low);
        if (old == high) {
            return;
        }
    }
}

__global__ void guf_union_equal_edges(
    const std::uint8_t* image,
    int* parent,
    int width,
    int height
) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const int index = y * width + x;
    const std::uint8_t value = image[index];
    if (x > 0 && image[index - 1] == value) {
        guf_union_minroot(parent, index, index - 1);
    }
    if (y > 0) {
        if (image[index - width] == value) {
            guf_union_minroot(parent, index, index - width);
        }
        if (x > 0 && image[index - width - 1] == value) {
            guf_union_minroot(parent, index, index - width - 1);
        }
        if (x + 1 < width && image[index - width + 1] == value) {
            guf_union_minroot(parent, index, index - width + 1);
        }
    }
}

__global__ void guf_full_compress(int* parent, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) {
        return;
    }
    int root = parent[index];
    while (root != parent[root]) {
        root = parent[root];
    }
    parent[index] = root;
}

// ---------------------------------------------------------------------------
// Pixel-level FlatZone-OLE CUDA route
// ---------------------------------------------------------------------------

__global__ void ole_init_labels(int* labels, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        labels[index] = index + 1;
    }
}

__global__ void ole_scan_equal(
    const std::uint8_t* image,
    int* labels,
    int width,
    int height,
    int* changed
) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const int index = y * width + x;
    const std::uint8_t value = image[index];
    const int label = labels[index];
    int minimum_label = label;
    for (int dy = -1; dy <= 1; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= height) {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            const int nx = x + dx;
            if (nx < 0 || nx >= width) {
                continue;
            }
            const int neighbour = ny * width + nx;
            if (image[neighbour] == value) {
                minimum_label = min(minimum_label, labels[neighbour]);
            }
        }
    }
    if (minimum_label < label) {
        const int old = atomicMin(labels + (label - 1), minimum_label);
        if (minimum_label < old) {
            atomicExch(changed, 1);
        }
    }
}

__global__ void ole_analyze_one_based(int* labels, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) {
        return;
    }
    int label = labels[index];
    int cursor = index;
    while (label - 1 != cursor) {
        cursor = label - 1;
        label = labels[cursor];
    }
    labels[index] = label;
}

__global__ void ole_convert_to_zero_based(int* labels, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        labels[index] -= 1;
    }
}

static void clear_root_flags(DeviceBuffers& buffers, std::size_t uint_bytes) {
    CUDA_CHECK(cudaMemset(buffers.root_any_greater, 0, uint_bytes));
    CUDA_CHECK(cudaMemset(buffers.root_any_smaller, 0, uint_bytes));
}

static int run_global_uf(
    DeviceBuffers& buffers,
    int width,
    int height,
    int block
) {
    const int count = width * height;
    const int linear_threads = 256;
    const int linear_blocks = (count + linear_threads - 1) / linear_threads;
    const dim3 block_2d(block, block);
    const dim3 grid_2d(
        (width + block - 1) / block,
        (height + block - 1) / block);
    const std::size_t uint_bytes =
        static_cast<std::size_t>(count) * sizeof(std::uint32_t);

    clear_root_flags(buffers, uint_bytes);
    guf_init_parent<<<linear_blocks, linear_threads>>>(buffers.labels, count);
    common_compute_boundary_flags<<<grid_2d, block_2d>>>(
        buffers.image,
        buffers.has_greater,
        buffers.has_smaller,
        width,
        height);
    guf_union_equal_edges<<<grid_2d, block_2d>>>(
        buffers.image, buffers.labels, width, height);
    guf_full_compress<<<linear_blocks, linear_threads>>>(
        buffers.labels, count);
    common_reduce_flags<<<linear_blocks, linear_threads>>>(
        buffers.labels,
        buffers.has_greater,
        buffers.has_smaller,
        buffers.root_any_greater,
        buffers.root_any_smaller,
        count);
    common_materialize_masks<<<linear_blocks, linear_threads>>>(
        buffers.labels,
        buffers.root_any_greater,
        buffers.root_any_smaller,
        buffers.maxima,
        buffers.minima,
        count);
    CUDA_CHECK(cudaGetLastError());
    return 1;
}

static int run_flatzone_ole(
    DeviceBuffers& buffers,
    int width,
    int height,
    int block
) {
    const int count = width * height;
    const int linear_threads = 256;
    const int linear_blocks = (count + linear_threads - 1) / linear_threads;
    const dim3 block_2d(block, block);
    const dim3 grid_2d(
        (width + block - 1) / block,
        (height + block - 1) / block);
    const std::size_t uint_bytes =
        static_cast<std::size_t>(count) * sizeof(std::uint32_t);

    clear_root_flags(buffers, uint_bytes);
    ole_init_labels<<<linear_blocks, linear_threads>>>(buffers.labels, count);
    int changed = 0;
    int iterations = 0;
    do {
        changed = 0;
        CUDA_CHECK(cudaMemcpy(
            buffers.changed,
            &changed,
            sizeof(int),
            cudaMemcpyHostToDevice));
        ole_scan_equal<<<grid_2d, block_2d>>>(
            buffers.image,
            buffers.labels,
            width,
            height,
            buffers.changed);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(
            &changed,
            buffers.changed,
            sizeof(int),
            cudaMemcpyDeviceToHost));
        if (changed != 0) {
            ole_analyze_one_based<<<linear_blocks, linear_threads>>>(
                buffers.labels, count);
        }
        ++iterations;
        if (iterations > count + 1) {
            throw std::runtime_error("OLE iteration limit exceeded");
        }
    } while (changed != 0);

    ole_analyze_one_based<<<linear_blocks, linear_threads>>>(
        buffers.labels, count);
    ole_convert_to_zero_based<<<linear_blocks, linear_threads>>>(
        buffers.labels, count);
    common_compute_boundary_flags<<<grid_2d, block_2d>>>(
        buffers.image,
        buffers.has_greater,
        buffers.has_smaller,
        width,
        height);
    common_reduce_flags<<<linear_blocks, linear_threads>>>(
        buffers.labels,
        buffers.has_greater,
        buffers.has_smaller,
        buffers.root_any_greater,
        buffers.root_any_smaller,
        count);
    common_materialize_masks<<<linear_blocks, linear_threads>>>(
        buffers.labels,
        buffers.root_any_greater,
        buffers.root_any_smaller,
        buffers.maxima,
        buffers.minima,
        count);
    CUDA_CHECK(cudaGetLastError());
    return iterations;
}

static void allocate_buffers(DeviceBuffers& buffers, int count) {
    const std::size_t bytes =
        static_cast<std::size_t>(count) * sizeof(std::uint8_t);
    const std::size_t int_bytes =
        static_cast<std::size_t>(count) * sizeof(int);
    const std::size_t uint_bytes =
        static_cast<std::size_t>(count) * sizeof(std::uint32_t);
    CUDA_CHECK(cudaMalloc(&buffers.image, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.labels, int_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.has_greater, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.has_smaller, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.root_any_greater, uint_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.root_any_smaller, uint_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.maxima, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.minima, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.changed, sizeof(int)));
}

static void free_buffers(DeviceBuffers& buffers) {
    CUDA_CHECK(cudaFree(buffers.image));
    CUDA_CHECK(cudaFree(buffers.labels));
    CUDA_CHECK(cudaFree(buffers.has_greater));
    CUDA_CHECK(cudaFree(buffers.has_smaller));
    CUDA_CHECK(cudaFree(buffers.root_any_greater));
    CUDA_CHECK(cudaFree(buffers.root_any_smaller));
    CUDA_CHECK(cudaFree(buffers.maxima));
    CUDA_CHECK(cudaFree(buffers.minima));
    CUDA_CHECK(cudaFree(buffers.changed));
    buffers = DeviceBuffers{};
}

static double percentile(std::vector<double> values, double probability) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position = probability * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

using AlgorithmFunction = int (*)(DeviceBuffers&, int, int, int);

static TimingSummary benchmark_algorithm(
    AlgorithmFunction algorithm,
    DeviceBuffers& buffers,
    const cv::Mat& image,
    int block,
    int warmup,
    int repetitions,
    cv::Mat& last_maxima,
    cv::Mat& last_minima
) {
    const int width = image.cols;
    const int height = image.rows;
    const int count = width * height;
    const std::size_t bytes =
        static_cast<std::size_t>(count) * sizeof(std::uint8_t);
    CUDA_CHECK(cudaMemcpy(
        buffers.image, image.data, bytes, cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup; ++i) {
        algorithm(buffers, width, height, block);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> compute_times;
    std::vector<double> h2h_times;
    std::vector<double> iterations;
    compute_times.reserve(static_cast<std::size_t>(repetitions));
    h2h_times.reserve(static_cast<std::size_t>(repetitions));
    iterations.reserve(static_cast<std::size_t>(repetitions));

    for (int rep = 0; rep < repetitions; ++rep) {
        CUDA_CHECK(cudaEventRecord(start));
        const int count_iterations =
            algorithm(buffers, width, height, block);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        compute_times.push_back(static_cast<double>(elapsed_ms));
        iterations.push_back(static_cast<double>(count_iterations));
    }

    last_maxima.create(height, width, CV_8U);
    last_minima.create(height, width, CV_8U);
    for (int rep = 0; rep < repetitions; ++rep) {
        const auto begin = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMemcpy(
            buffers.image, image.data, bytes, cudaMemcpyHostToDevice));
        const int count_iterations =
            algorithm(buffers, width, height, block);
        CUDA_CHECK(cudaMemcpy(
            last_maxima.data,
            buffers.maxima,
            bytes,
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            last_minima.data,
            buffers.minima,
            bytes,
            cudaMemcpyDeviceToHost));
        const auto end = std::chrono::high_resolution_clock::now();
        h2h_times.push_back(
            std::chrono::duration<double, std::milli>(end - begin).count());
        iterations.push_back(static_cast<double>(count_iterations));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    TimingSummary summary;
    summary.median_compute_ms = percentile(compute_times, 0.5);
    summary.p95_compute_ms = percentile(compute_times, 0.95);
    summary.median_h2h_ms = percentile(h2h_times, 0.5);
    summary.p95_h2h_ms = percentile(h2h_times, 0.95);
    summary.median_iterations = percentile(iterations, 0.5);
    summary.p95_iterations = percentile(iterations, 0.95);
    return summary;
}

static std::uint64_t compare_masks(
    const cv::Mat& first,
    const cv::Mat& second
) {
    if (first.rows != second.rows || first.cols != second.cols) {
        throw std::runtime_error("Cross-check dimensions differ");
    }
    std::uint64_t mismatches = 0;
    const int count = first.rows * first.cols;
    for (int i = 0; i < count; ++i) {
        if ((first.data[i] != 0) != (second.data[i] != 0)) {
            ++mismatches;
        }
    }
    return mismatches;
}

static std::string quote_csv(const std::string& value) {
    std::string escaped = "\"";
    for (const char c : value) {
        if (c == '"') {
            escaped += "\"\"";
        } else {
            escaped += c;
        }
    }
    escaped += '"';
    return escaped;
}

static void write_row(
    std::ofstream& csv,
    const std::string& algorithm,
    const fs::path& image_path,
    int width,
    int height,
    int warmup,
    int repetitions,
    int block,
    std::uint64_t allocated_bytes,
    const TimingSummary& timing,
    std::uint64_t max_mismatches,
    std::uint64_t min_mismatches
) {
    const double megapixels =
        static_cast<double>(width) * static_cast<double>(height) / 1e6;
    const double compute_throughput =
        timing.median_compute_ms > 0.0
            ? megapixels / (timing.median_compute_ms / 1000.0)
            : 0.0;
    const double h2h_throughput =
        timing.median_h2h_ms > 0.0
            ? megapixels / (timing.median_h2h_ms / 1000.0)
            : 0.0;
    csv << quote_csv(algorithm) << ','
        << quote_csv(image_path.filename().string()) << ','
        << width << ',' << height << ',' << std::fixed << std::setprecision(6)
        << megapixels << ',' << warmup << ',' << repetitions << ',' << block
        << ',' << allocated_bytes << ','
        << static_cast<double>(allocated_bytes) /
               static_cast<double>(width * height)
        << ',' << timing.median_compute_ms << ',' << timing.p95_compute_ms
        << ',' << compute_throughput << ',' << timing.median_h2h_ms << ','
        << timing.p95_h2h_ms << ',' << h2h_throughput << ','
        << timing.median_iterations << ',' << timing.p95_iterations << ','
        << max_mismatches << ',' << min_mismatches << '\n';
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "Usage: gpu_benchmark_globaluf_ole_v0_1.exe <image_dir>"
               " --output=DIR [--device=0] [--block=16]"
               " [--warmup=5] [--reps=30]\n";
        return 2;
    }

    try {
        const fs::path image_dir = argv[1];
        fs::path output_dir;
        int device = 0;
        int block = 16;
        int warmup = 5;
        int repetitions = 30;
        for (int i = 2; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument.rfind("--output=", 0) == 0) {
                output_dir = argument.substr(9);
            } else if (argument.rfind("--device=", 0) == 0) {
                device = std::stoi(argument.substr(9));
            } else if (argument.rfind("--block=", 0) == 0) {
                block = std::stoi(argument.substr(8));
            } else if (argument.rfind("--warmup=", 0) == 0) {
                warmup = std::stoi(argument.substr(9));
            } else if (argument.rfind("--reps=", 0) == 0) {
                repetitions = std::stoi(argument.substr(7));
            }
        }
        if (output_dir.empty()) {
            throw std::runtime_error("Missing --output directory");
        }
        if (block < 4 || block > 32 || warmup < 0 || repetitions < 1) {
            throw std::runtime_error("Invalid block/warmup/repetition value");
        }

        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        fs::create_directories(output_dir);
        std::ofstream csv(output_dir / "gpu_common_benchmark.csv");
        if (!csv) {
            throw std::runtime_error("Cannot create benchmark CSV");
        }
        csv << "algorithm,image,width,height,megapixels,warmup,repetitions,"
               "block,allocated_bytes,allocated_bytes_per_pixel,"
               "median_compute_ms,p95_compute_ms,compute_mpix_s,"
               "median_h2h_ms,p95_h2h_ms,h2h_mpix_s,"
               "median_iterations,p95_iterations,"
               "cross_max_mismatches,cross_min_mismatches\n";

        std::ofstream metadata(output_dir / "environment.txt");
        metadata << "device_name=" << properties.name << '\n'
                 << "compute_capability=" << properties.major << '.'
                 << properties.minor << '\n'
                 << "global_memory_bytes=" << properties.totalGlobalMem << '\n'
                 << "cuda_runtime=" << CUDART_VERSION << '\n'
                 << "block=" << block << '\n'
                 << "warmup=" << warmup << '\n'
                 << "repetitions=" << repetitions << '\n'
                 << "timing_compute=preallocated,input resident,full masks\n"
                 << "timing_h2h=reused allocation,H2D+compute+D2H two masks\n";

        const std::vector<fs::path> files = list_images(image_dir);
        std::uint64_t total_max_mismatches = 0;
        std::uint64_t total_min_mismatches = 0;
        for (const fs::path& image_path : files) {
            cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
            if (image.empty()) {
                throw std::runtime_error("Cannot read " + image_path.string());
            }
            if (!image.isContinuous()) {
                image = image.clone();
            }
            const int width = image.cols;
            const int height = image.rows;
            const int count = width * height;
            DeviceBuffers buffers;
            allocate_buffers(buffers, count);

            cv::Mat guf_max;
            cv::Mat guf_min;
            cv::Mat ole_max;
            cv::Mat ole_min;
            const TimingSummary guf_timing = benchmark_algorithm(
                run_global_uf,
                buffers,
                image,
                block,
                warmup,
                repetitions,
                guf_max,
                guf_min);
            const TimingSummary ole_timing = benchmark_algorithm(
                run_flatzone_ole,
                buffers,
                image,
                block,
                warmup,
                repetitions,
                ole_max,
                ole_min);

            const std::uint64_t max_mismatches =
                compare_masks(guf_max, ole_max);
            const std::uint64_t min_mismatches =
                compare_masks(guf_min, ole_min);
            total_max_mismatches += max_mismatches;
            total_min_mismatches += min_mismatches;
            const std::uint64_t common_bytes =
                static_cast<std::uint64_t>(count) * 17ull;

            write_row(
                csv,
                "Global-UF-CUDA",
                image_path,
                width,
                height,
                warmup,
                repetitions,
                block,
                common_bytes,
                guf_timing,
                max_mismatches,
                min_mismatches);
            write_row(
                csv,
                "FlatZone-OLE-CUDA",
                image_path,
                width,
                height,
                warmup,
                repetitions,
                block,
                common_bytes + sizeof(int),
                ole_timing,
                max_mismatches,
                min_mismatches);
            csv.flush();

            std::cout << "[GPU-BENCH] " << image_path.filename().string()
                      << " " << width << 'x' << height
                      << " GUF_compute_ms=" << guf_timing.median_compute_ms
                      << " OLE_compute_ms=" << ole_timing.median_compute_ms
                      << " OLE_iterations=" << ole_timing.median_iterations
                      << " max_mismatch=" << max_mismatches
                      << " min_mismatch=" << min_mismatches << '\n';
            free_buffers(buffers);
        }

        std::cout << "[GPU-COMMON-BENCHMARK-OK] images=" << files.size()
                  << " max_mismatches=" << total_max_mismatches
                  << " min_mismatches=" << total_min_mismatches
                  << " output=" << output_dir.string() << '\n';
        return total_max_mismatches == 0 && total_min_mismatches == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 3;
    }
}
