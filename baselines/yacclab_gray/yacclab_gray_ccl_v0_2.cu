// Task-complete grayscale adaptations of two pixel-based GPU CCL algorithms
// distributed with YACCLAB (BSD 3-Clause), for exact 8-connected dual
// regional-extrema masks, v0.2.
//
//   UF-Gray: the union-find algorithm of Oliveira and Lotufo (SIBGRAPI 2010) as
//            implemented in YACCLAB (labeling_oliveira_2010.cu): shared-memory
//            union-find inside 16x16 blocks, global union-find on block borders,
//            path compression.
//   KE-Gray: the 8-connected Komura-equivalence algorithm of Allegretti et al.
//            as implemented in YACCLAB (labeling_allegretti_2018.cu): init to a
//            preceding equal neighbour, compression, reduction of the left and
//            up-right edges, compression.
//
// Grayscale adaptation: every pixel is labelled, and "both pixels are
// foreground" becomes "both pixels have the same value". Both algorithms rely
// only on the transitivity of the foreground relation to skip edges; equality
// is transitive as well, so every equal-valued 8-neighbour edge is still
// united. UF's binary trick of letting a background pixel unite its left (or
// right) and up neighbours becomes: a pixel whose up neighbour differs from it
// unites that up neighbour with an equal left (or right) neighbour. Pixels
// outside the image are treated as absent, never as a value.
//
// After labelling, both routes share the classification and output stages of
// the dense union-find reference: per-pixel greater/smaller flags over the
// valid 8-neighbourhood, atomic OR at the component root, and materialization
// of both masks. Labels are 1-based (root index + 1) as in YACCLAB.
//
// v0.2 adds --levels=N: the input is quantized after loading, outside the
// timed region, with the rule of the quantization harness
// (bin = floor(value * levels / 256), output = round(bin * 255 / (levels - 1))),
// so both harnesses time identical images. --levels=256 leaves the input
// unchanged and reproduces v0.1.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t code = (call);                                       \
        if (code != cudaSuccess) {                                             \
            throw std::runtime_error(                                          \
                std::string("CUDA error: ") + cudaGetErrorString(code));       \
        }                                                                      \
    } while (0)

constexpr int BLOCK_ROWS = 16;
constexpr int BLOCK_COLS = 16;

struct Buffers {
    std::uint8_t* image = nullptr;
    int* labels = nullptr;
    std::uint8_t* has_greater = nullptr;
    std::uint8_t* has_smaller = nullptr;
    std::uint32_t* root_any_greater = nullptr;
    std::uint32_t* root_any_smaller = nullptr;
    std::uint8_t* maxima = nullptr;
    std::uint8_t* minima = nullptr;
};

struct Timing {
    double median_ms = 0.0;
    double p95_ms = 0.0;
};

// ---------------------------------------------------------------------------
// Union-find on 1-based labels (YACCLAB convention)
// ---------------------------------------------------------------------------

__device__ unsigned find_root(const int* buffer, unsigned node) {
    unsigned label = buffer[node];
    while (label - 1 != node) {
        node = label - 1;
        label = buffer[node];
    }
    return node;
}

__device__ unsigned find_root_from(const int* buffer, unsigned node,
                                   unsigned label) {
    while (label - 1 != node) {
        node = label - 1;
        label = buffer[node];
    }
    return node;
}

__device__ void unite(int* buffer, unsigned a, unsigned b) {
    bool done;
    do {
        a = find_root(buffer, a);
        b = find_root(buffer, b);
        if (a < b) {
            const int old = atomicMin(buffer + b, static_cast<int>(a + 1));
            done = (old == static_cast<int>(b + 1));
            b = old - 1;
        } else if (b < a) {
            const int old = atomicMin(buffer + a, static_cast<int>(b + 1));
            done = (old == static_cast<int>(a + 1));
            a = old - 1;
        } else {
            done = true;
        }
    } while (!done);
}

// ---------------------------------------------------------------------------
// UF-Gray (Oliveira and Lotufo 2010, YACCLAB)
// ---------------------------------------------------------------------------

__global__ void uf_local_merge(const std::uint8_t* image, int* labels,
                               int width, int height) {
    const int local_row = threadIdx.y;
    const int local_col = threadIdx.x;
    const int local_index = local_row * BLOCK_COLS + local_col;
    const int row = blockIdx.y * BLOCK_ROWS + local_row;
    const int col = blockIdx.x * BLOCK_COLS + local_col;

    __shared__ int s_buf[BLOCK_ROWS * BLOCK_COLS];
    __shared__ std::uint8_t s_img[BLOCK_ROWS * BLOCK_COLS];

    const bool inside = row < height && col < width;
    s_buf[local_index] = local_index + 1;
    s_img[local_index] = inside ? image[row * width + col] : 0;
    __syncthreads();

    if (inside) {
        const std::uint8_t value = s_img[local_index];
        const bool has_left = local_col > 0;
        const bool has_up = local_row > 0;
        const bool has_right = local_col < BLOCK_COLS - 1 && col + 1 < width;
        if (has_left && s_img[local_index - 1] == value) {
            unite(s_buf, local_index, local_index - 1);
        }
        if (has_up) {
            const std::uint8_t up = s_img[local_index - BLOCK_COLS];
            if (up == value) {
                unite(s_buf, local_index, local_index - BLOCK_COLS);
            } else {
                if (has_left && s_img[local_index - 1] == up) {
                    unite(s_buf, local_index - 1, local_index - BLOCK_COLS);
                }
                if (has_right && s_img[local_index + 1] == up) {
                    unite(s_buf, local_index + 1, local_index - BLOCK_COLS);
                }
            }
        }
    }
    __syncthreads();

    if (inside) {
        const unsigned root = find_root(s_buf, local_index);
        const int root_row = blockIdx.y * BLOCK_ROWS + root / BLOCK_COLS;
        const int root_col = blockIdx.x * BLOCK_COLS + root % BLOCK_COLS;
        labels[row * width + col] = root_row * width + root_col + 1;
    }
}

__global__ void uf_global_merge(const std::uint8_t* image, int* labels,
                                int width, int height) {
    const int local_row = threadIdx.y;
    const int local_col = threadIdx.x;
    const int row = blockIdx.y * BLOCK_ROWS + local_row;
    const int col = blockIdx.x * BLOCK_COLS + local_col;
    if (row >= height || col >= width) {
        return;
    }
    const int index = row * width + col;
    const std::uint8_t value = image[index];
    if (col > 0 && local_col == 0 && image[index - 1] == value) {
        unite(labels, index, index - 1);
    }
    if (row > 0) {
        const std::uint8_t up = image[index - width];
        if (up == value) {
            if (local_row == 0) {
                unite(labels, index, index - width);
            }
        } else {
            if (col > 0 && (local_row == 0 || local_col == 0) &&
                image[index - 1] == up) {
                unite(labels, index - 1, index - width);
            }
            if (col + 1 < width &&
                (local_row == 0 || local_col == BLOCK_COLS - 1) &&
                image[index + 1] == up) {
                unite(labels, index + 1, index - width);
            }
        }
    }
}

__global__ void uf_path_compression(int* labels, int width, int height) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        const int index = row * width + col;
        labels[index] = find_root(labels, index) + 1;
    }
}

// ---------------------------------------------------------------------------
// KE-Gray (Allegretti et al., YACCLAB)
// ---------------------------------------------------------------------------

__global__ void ke_init(const std::uint8_t* image, int* labels, int width,
                        int height) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height || col >= width) {
        return;
    }
    const int index = row * width + col;
    const std::uint8_t value = image[index];
    int label = index + 1;
    if (row > 0 && image[index - width] == value) {
        label = index - width + 1;
    } else if (row > 0 && col > 0 && image[index - width - 1] == value) {
        label = index - width;
    } else if (row > 0 && col + 1 < width &&
               image[index - width + 1] == value) {
        label = index - width + 2;
    } else if (col > 0 && image[index - 1] == value) {
        label = index;
    }
    labels[index] = label;
}

__global__ void ke_compression(int* labels, int width, int height) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        const int index = row * width + col;
        labels[index] = find_root_from(labels, index, labels[index]) + 1;
    }
}

__global__ void ke_reduce(const std::uint8_t* image, int* labels, int width,
                          int height) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height || col >= width) {
        return;
    }
    const int index = row * width + col;
    const std::uint8_t value = image[index];
    if (col > 0 && image[index - 1] == value) {
        unite(labels, index, index - 1);
    }
    if (row > 0 && col + 1 < width && image[index - width + 1] == value) {
        unite(labels, index, index - width + 1);
    }
}

// ---------------------------------------------------------------------------
// Shared classification and output stages
// ---------------------------------------------------------------------------

__global__ void compute_boundary_flags(const std::uint8_t* image,
                                       std::uint8_t* has_greater,
                                       std::uint8_t* has_smaller, int width,
                                       int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
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

__global__ void reduce_flags(const int* labels_one_based,
                             const std::uint8_t* has_greater,
                             const std::uint8_t* has_smaller,
                             std::uint32_t* root_any_greater,
                             std::uint32_t* root_any_smaller, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int root = labels_one_based[index] - 1;
    if (has_greater[index] != 0) {
        atomicOr(root_any_greater + root, 1u);
    }
    if (has_smaller[index] != 0) {
        atomicOr(root_any_smaller + root, 1u);
    }
}

__global__ void materialize_masks(const int* labels_one_based,
                                  const std::uint32_t* root_any_greater,
                                  const std::uint32_t* root_any_smaller,
                                  std::uint8_t* maxima, std::uint8_t* minima,
                                  int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int root = labels_one_based[index] - 1;
    maxima[index] = root_any_greater[root] == 0 ? 255 : 0;
    minima[index] = root_any_smaller[root] == 0 ? 255 : 0;
}

// ---------------------------------------------------------------------------
// Task-complete routes
// ---------------------------------------------------------------------------

enum class Route { UF, KE };

static const char* route_name(Route route) {
    return route == Route::UF ? "UF-Gray-YACCLAB" : "KE-Gray-YACCLAB";
}

static void run_task_complete(Route route, Buffers& buffers, int width,
                              int height) {
    const int count = width * height;
    const int linear_threads = 256;
    const int linear_blocks = (count + linear_threads - 1) / linear_threads;
    const dim3 block(BLOCK_COLS, BLOCK_ROWS);
    const dim3 grid((width + BLOCK_COLS - 1) / BLOCK_COLS,
                    (height + BLOCK_ROWS - 1) / BLOCK_ROWS);
    const std::size_t uint_bytes =
        static_cast<std::size_t>(count) * sizeof(std::uint32_t);

    CUDA_CHECK(cudaMemset(buffers.root_any_greater, 0, uint_bytes));
    CUDA_CHECK(cudaMemset(buffers.root_any_smaller, 0, uint_bytes));
    compute_boundary_flags<<<grid, block>>>(
        buffers.image, buffers.has_greater, buffers.has_smaller, width,
        height);
    if (route == Route::UF) {
        uf_local_merge<<<grid, block>>>(buffers.image, buffers.labels, width,
                                        height);
        uf_global_merge<<<grid, block>>>(buffers.image, buffers.labels, width,
                                         height);
        uf_path_compression<<<grid, block>>>(buffers.labels, width, height);
    } else {
        ke_init<<<grid, block>>>(buffers.image, buffers.labels, width, height);
        ke_compression<<<grid, block>>>(buffers.labels, width, height);
        ke_reduce<<<grid, block>>>(buffers.image, buffers.labels, width,
                                   height);
        ke_compression<<<grid, block>>>(buffers.labels, width, height);
    }
    reduce_flags<<<linear_blocks, linear_threads>>>(
        buffers.labels, buffers.has_greater, buffers.has_smaller,
        buffers.root_any_greater, buffers.root_any_smaller, count);
    materialize_masks<<<linear_blocks, linear_threads>>>(
        buffers.labels, buffers.root_any_greater, buffers.root_any_smaller,
        buffers.maxima, buffers.minima, count);
    CUDA_CHECK(cudaGetLastError());
}

static void allocate_buffers(Buffers& buffers, int pixels) {
    const std::size_t n = static_cast<std::size_t>(pixels);
    CUDA_CHECK(cudaMalloc(&buffers.image, n));
    CUDA_CHECK(cudaMalloc(&buffers.labels, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers.has_greater, n));
    CUDA_CHECK(cudaMalloc(&buffers.has_smaller, n));
    CUDA_CHECK(cudaMalloc(&buffers.root_any_greater, n * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&buffers.root_any_smaller, n * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&buffers.maxima, n));
    CUDA_CHECK(cudaMalloc(&buffers.minima, n));
}

static void free_buffers(Buffers& buffers) {
    cudaFree(buffers.image);
    cudaFree(buffers.labels);
    cudaFree(buffers.has_greater);
    cudaFree(buffers.has_smaller);
    cudaFree(buffers.root_any_greater);
    cudaFree(buffers.root_any_smaller);
    cudaFree(buffers.maxima);
    cudaFree(buffers.minima);
    buffers = Buffers{};
}

static double percentile(std::vector<double> values, double probability) {
    std::sort(values.begin(), values.end());
    const double position =
        probability * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

static Timing benchmark(Route route, Buffers& buffers, int width, int height,
                        int warmup, int repetitions, cv::Mat& maxima,
                        cv::Mat& minima) {
    for (int i = 0; i < warmup; ++i) {
        run_task_complete(route, buffers, width, height);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start;
    cudaEvent_t finish;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&finish));
    std::vector<double> elapsed;
    for (int i = 0; i < repetitions; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        run_task_complete(route, buffers, width, height);
        CUDA_CHECK(cudaEventRecord(finish));
        CUDA_CHECK(cudaEventSynchronize(finish));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, finish));
        elapsed.push_back(milliseconds);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(finish));
    const int pixels = width * height;
    maxima.create(height, width, CV_8UC1);
    minima.create(height, width, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(maxima.data, buffers.maxima, pixels,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(minima.data, buffers.minima, pixels,
                          cudaMemcpyDeviceToHost));
    return Timing{percentile(elapsed, 0.5), percentile(elapsed, 0.95)};
}

static void quantize_grayscale_in_place(cv::Mat& image, int levels) {
    if (levels == 256) {
        return;
    }
    const int count = image.rows * image.cols;
    for (int index = 0; index < count; ++index) {
        const int value = static_cast<int>(image.data[index]);
        const int bin = (value * levels) >> 8;
        image.data[index] = static_cast<std::uint8_t>(
            (bin * 255 + (levels - 1) / 2) / (levels - 1));
    }
}

static bool ends_with_ci(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) {
        return false;
    }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        const char a = static_cast<char>(std::tolower(static_cast<unsigned char>(
            value[value.size() - suffix.size() + i])));
        const char b = static_cast<char>(
            std::tolower(static_cast<unsigned char>(suffix[i])));
        if (a != b) {
            return false;
        }
    }
    return true;
}

static std::vector<fs::path> list_images(const fs::path& directory) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (ends_with_ci(name, ".png") || ends_with_ci(name, ".pgm") ||
            ends_with_ci(name, ".jpg") || ends_with_ci(name, ".jpeg") ||
            ends_with_ci(name, ".bmp") || ends_with_ci(name, ".tif") ||
            ends_with_ci(name, ".tiff")) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::runtime_error("No images found in " + directory.string());
    }
    return files;
}

static std::uint64_t compare_masks(const cv::Mat& first, const cv::Mat& second) {
    if (first.rows != second.rows || first.cols != second.cols) {
        throw std::runtime_error("Mask dimensions differ");
    }
    std::uint64_t mismatches = 0;
    for (int y = 0; y < first.rows; ++y) {
        const auto* a = first.ptr<std::uint8_t>(y);
        const auto* b = second.ptr<std::uint8_t>(y);
        for (int x = 0; x < first.cols; ++x) {
            mismatches += (a[x] != 0) != (b[x] != 0) ? 1ull : 0ull;
        }
    }
    return mismatches;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: yacclab_gray_ccl_v0_2.exe <image_dir>"
                     " --output=DIR [--expected-max=DIR --expected-min=DIR]"
                     " [--device=0] [--warmup=5] [--reps=30] [--levels=256]\n";
        return 2;
    }
    try {
        const fs::path image_dir = argv[1];
        fs::path output_dir;
        fs::path expected_max_dir;
        fs::path expected_min_dir;
        int device = 0;
        int warmup = 5;
        int repetitions = 30;
        int levels = 256;
        for (int index = 2; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument.rfind("--output=", 0) == 0) {
                output_dir = argument.substr(9);
            } else if (argument.rfind("--expected-max=", 0) == 0) {
                expected_max_dir = argument.substr(15);
            } else if (argument.rfind("--expected-min=", 0) == 0) {
                expected_min_dir = argument.substr(15);
            } else if (argument.rfind("--device=", 0) == 0) {
                device = std::stoi(argument.substr(9));
            } else if (argument.rfind("--warmup=", 0) == 0) {
                warmup = std::stoi(argument.substr(9));
            } else if (argument.rfind("--reps=", 0) == 0) {
                repetitions = std::stoi(argument.substr(7));
            } else if (argument.rfind("--levels=", 0) == 0) {
                levels = std::stoi(argument.substr(9));
            } else {
                throw std::runtime_error("Unknown argument: " + argument);
            }
        }
        if (output_dir.empty()) {
            throw std::runtime_error("Missing --output directory");
        }
        if (expected_max_dir.empty() != expected_min_dir.empty()) {
            throw std::runtime_error(
                "Expected max/min directories must be supplied together");
        }
        if (warmup < 0 || repetitions < 1) {
            throw std::runtime_error("Invalid benchmark option");
        }
        if (levels < 2 || levels > 256 || (levels & (levels - 1)) != 0) {
            throw std::runtime_error(
                "Quantization levels must be a power of two in [2,256]");
        }
        if (levels != 256 && !expected_max_dir.empty()) {
            throw std::runtime_error(
                "Oracle masks refer to unquantized inputs; use --levels=256");
        }
        if (fs::exists(output_dir)) {
            throw std::runtime_error(
                "Refusing to overwrite existing output directory: " +
                output_dir.string());
        }

        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        fs::create_directories(output_dir);
        std::ofstream csv(output_dir / "yacclab_gray_ccl_v0_2.csv");
        csv << "algorithm,image,quantization_levels,width,height,megapixels,"
               "warmup,repetitions,"
               "total_allocated_bytes,total_allocated_Bpp,"
               "auxiliary_bytes,auxiliary_Bpp,"
               "median_compute_ms,p95_compute_ms,compute_mpix_s,"
               "oracle_max_mismatches,oracle_min_mismatches,"
               "cross_route_mismatches\n";
        std::ofstream environment(output_dir / "environment.txt");
        environment << "harness=yacclab_gray_ccl_v0.2\n"
                    << "routes=UF-Gray(Oliveira-Lotufo 2010),"
                       "KE-Gray(Allegretti et al.)\n"
                    << "source=YACCLAB labeling_oliveira_2010.cu, "
                       "labeling_allegretti_2018.cu (BSD 3-Clause)\n"
                    << "device_name=" << properties.name << '\n'
                    << "connectivity=8\n"
                    << "block=" << BLOCK_COLS << "x" << BLOCK_ROWS << '\n'
                    << "timing=resident_input_preallocated_task_complete\n"
                    << "memory=input+two_outputs+int_labels+two_byte_flags+"
                       "two_uint_root_flags\n"
                    << "warmup=" << warmup << '\n'
                    << "repetitions=" << repetitions << '\n'
                    << "quantization_levels=" << levels << '\n'
                    << "quantization_rule=bin=floor(value*levels/256);"
                       "output=round(bin*255/(levels-1));conversion_not_timed\n";

        const auto files = list_images(image_dir);
        std::uint64_t total_mismatches = 0;
        double total_megapixels = 0.0;
        double total_seconds[2] = {0.0, 0.0};
        for (const fs::path& image_path : files) {
            cv::Mat image = cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
            if (image.empty()) {
                throw std::runtime_error("Cannot read " + image_path.string());
            }
            if (!image.isContinuous()) {
                image = image.clone();
            }
            quantize_grayscale_in_place(image, levels);
            const int width = image.cols;
            const int height = image.rows;
            const int pixels = width * height;
            Buffers buffers;
            allocate_buffers(buffers, pixels);
            CUDA_CHECK(cudaMemcpy(buffers.image, image.data, pixels,
                                  cudaMemcpyHostToDevice));

            cv::Mat expected_max;
            cv::Mat expected_min;
            if (!expected_max_dir.empty()) {
                const std::string mask_name = image_path.stem().string() + ".pgm";
                expected_max = cv::imread((expected_max_dir / mask_name).string(),
                                          cv::IMREAD_GRAYSCALE);
                expected_min = cv::imread((expected_min_dir / mask_name).string(),
                                          cv::IMREAD_GRAYSCALE);
                if (expected_max.empty() || expected_min.empty()) {
                    throw std::runtime_error("Missing oracle masks for " +
                                             image_path.filename().string());
                }
            }

            const double megapixels = static_cast<double>(pixels) / 1e6;
            total_megapixels += megapixels;
            cv::Mat masks[2][2];
            Timing timings[2];
            const Route routes[2] = {Route::UF, Route::KE};
            for (int r = 0; r < 2; ++r) {
                timings[r] = benchmark(routes[r], buffers, width, height, warmup,
                                       repetitions, masks[r][0], masks[r][1]);
                total_seconds[r] += timings[r].median_ms / 1000.0;
            }
            const std::uint64_t cross = compare_masks(masks[0][0], masks[1][0]) +
                                        compare_masks(masks[0][1], masks[1][1]);
            total_mismatches += cross;

            const std::uint64_t total_bytes =
                static_cast<std::uint64_t>(pixels) * 17ull;
            const std::uint64_t auxiliary_bytes =
                static_cast<std::uint64_t>(pixels) * 14ull;
            std::cout << "[YACCLAB-GRAY-v0.2] " << image_path.filename().string();
            for (int r = 0; r < 2; ++r) {
                std::uint64_t max_mismatches = 0;
                std::uint64_t min_mismatches = 0;
                if (!expected_max.empty()) {
                    max_mismatches = compare_masks(masks[r][0], expected_max);
                    min_mismatches = compare_masks(masks[r][1], expected_min);
                    total_mismatches += max_mismatches + min_mismatches;
                }
                csv << '"' << route_name(routes[r]) << "\",\""
                    << image_path.filename().string() << "\"," << levels << ','
                    << width << ','
                    << height << ',' << std::fixed << std::setprecision(6)
                    << megapixels << ',' << warmup << ',' << repetitions << ','
                    << total_bytes << ','
                    << static_cast<double>(total_bytes) / pixels << ','
                    << auxiliary_bytes << ','
                    << static_cast<double>(auxiliary_bytes) / pixels << ','
                    << timings[r].median_ms << ',' << timings[r].p95_ms << ','
                    << megapixels / (timings[r].median_ms / 1000.0) << ','
                    << max_mismatches << ',' << min_mismatches << ',' << cross
                    << '\n';
                std::cout << ' ' << (r == 0 ? "UF_ms=" : "KE_ms=")
                          << timings[r].median_ms
                          << " oracle_mismatch=" << max_mismatches + min_mismatches;
            }
            std::cout << " cross_mismatch=" << cross << '\n';
            csv.flush();
            free_buffers(buffers);
        }
        std::cout << std::fixed << std::setprecision(3)
                  << (total_mismatches == 0 ? "[YACCLAB-GRAY-v0.2-OK]"
                                            : "[YACCLAB-GRAY-v0.2-FAIL]")
                  << " images=" << files.size()
                  << " UF_mpix_s=" << total_megapixels / total_seconds[0]
                  << " KE_mpix_s=" << total_megapixels / total_seconds[1]
                  << " mismatches=" << total_mismatches
                  << " output=" << output_dir.string() << '\n';
        return total_mismatches == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 3;
    }
}
