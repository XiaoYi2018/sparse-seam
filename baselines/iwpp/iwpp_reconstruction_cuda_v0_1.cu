// Reproduction baseline of Teodoro et al.'s GPU irregular-wavefront
// morphological reconstruction for exact 8-neighbour regional extrema.
//
// This implementation uses four GPU scan passes for initialization and
// round-based global queues with atomicMax propagation. Queue entries are
// deduplicated per round, preserving exact monotone convergence with capacity N.

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

struct Buffers {
    std::uint8_t* image = nullptr;
    std::uint8_t* maxima = nullptr;
    std::uint8_t* minima = nullptr;
    int* marker = nullptr;
    int* queue_a = nullptr;
    int* queue_b = nullptr;
    int* next_flag = nullptr;
    int* count = nullptr;
};

struct Timing {
    double median_ms = 0.0;
    double p95_ms = 0.0;
    double median_rounds = 0.0;
    double p95_rounds = 0.0;
};

__device__ __forceinline__ int mask_value(
    const std::uint8_t* image,
    int index,
    bool complement) {
    return complement ? 255 - static_cast<int>(image[index])
                      : static_cast<int>(image[index]);
}

__global__ void initialize_marker(
    const std::uint8_t* image,
    int* marker,
    int pixels,
    bool complement) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < pixels) {
        marker[index] = mask_value(image, index, complement) - 1;
    }
}

__global__ void scan_rows_forward(
    const std::uint8_t* image,
    int* marker,
    int width,
    int height,
    bool complement) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) {
        return;
    }
    const int base = y * width;
    for (int x = 1; x < width; ++x) {
        const int p = base + x;
        marker[p] = min(
            max(marker[p], marker[p - 1]),
            mask_value(image, p, complement));
    }
}

__global__ void scan_columns_forward(
    const std::uint8_t* image,
    int* marker,
    int width,
    int height,
    bool complement) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) {
        return;
    }
    for (int y = 1; y < height; ++y) {
        const int p = y * width + x;
        marker[p] = min(
            max(marker[p], marker[p - width]),
            mask_value(image, p, complement));
    }
}

__global__ void scan_rows_backward(
    const std::uint8_t* image,
    int* marker,
    int width,
    int height,
    bool complement) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= height) {
        return;
    }
    const int base = y * width;
    for (int x = width - 2; x >= 0; --x) {
        const int p = base + x;
        marker[p] = min(
            max(marker[p], marker[p + 1]),
            mask_value(image, p, complement));
    }
}

__global__ void scan_columns_backward(
    const std::uint8_t* image,
    int* marker,
    int width,
    int height,
    bool complement) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) {
        return;
    }
    for (int y = height - 2; y >= 0; --y) {
        const int p = y * width + x;
        marker[p] = min(
            max(marker[p], marker[p + width]),
            mask_value(image, p, complement));
    }
}

__global__ void seed_wavefront(
    const std::uint8_t* image,
    const int* marker,
    int* queue,
    int* count,
    int width,
    int height,
    bool complement) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixels = width * height;
    if (p >= pixels) {
        return;
    }
    const int y = p / width;
    const int x = p - y * width;
    const int jp = marker[p];
    bool source = false;
    for (int dy = -1; dy <= 1 && !source; ++dy) {
        const int qy = y + dy;
        if (qy < 0 || qy >= height) {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            const int qx = x + dx;
            if (qx < 0 || qx >= width) {
                continue;
            }
            const int q = qy * width + qx;
            if (marker[q] < jp &&
                marker[q] < mask_value(image, q, complement)) {
                source = true;
                break;
            }
        }
    }
    if (source) {
        const int position = atomicAdd(count, 1);
        queue[position] = p;
    }
}

__global__ void propagate_wavefront(
    const std::uint8_t* image,
    int* marker,
    const int* current_queue,
    int current_count,
    int* next_queue,
    int* next_count,
    int* next_flag,
    int width,
    int height,
    bool complement) {
    const int queue_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_index >= current_count) {
        return;
    }
    const int p = current_queue[queue_index];
    const int y = p / width;
    const int x = p - y * width;
    const int jp = marker[p];
    for (int dy = -1; dy <= 1; ++dy) {
        const int qy = y + dy;
        if (qy < 0 || qy >= height) {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) {
                continue;
            }
            const int qx = x + dx;
            if (qx < 0 || qx >= width) {
                continue;
            }
            const int q = qy * width + qx;
            const int candidate =
                min(jp, mask_value(image, q, complement));
            const int old_value = atomicMax(marker + q, candidate);
            if (old_value < candidate &&
                atomicCAS(next_flag + q, 0, 1) == 0) {
                const int position = atomicAdd(next_count, 1);
                next_queue[position] = q;
            }
        }
    }
}

__global__ void materialize_extrema(
    const std::uint8_t* image,
    const int* marker,
    std::uint8_t* output,
    int pixels,
    bool complement) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < pixels) {
        output[index] =
            mask_value(image, index, complement) > marker[index] ? 255 : 0;
    }
}

static void allocate_buffers(Buffers& buffers, int pixels) {
    CUDA_CHECK(cudaMalloc(&buffers.image, pixels));
    CUDA_CHECK(cudaMalloc(&buffers.maxima, pixels));
    CUDA_CHECK(cudaMalloc(&buffers.minima, pixels));
    CUDA_CHECK(cudaMalloc(&buffers.marker, pixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers.queue_a, pixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers.queue_b, pixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers.next_flag, pixels * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers.count, sizeof(int)));
}

static void free_buffers(Buffers& buffers) {
    cudaFree(buffers.image);
    cudaFree(buffers.maxima);
    cudaFree(buffers.minima);
    cudaFree(buffers.marker);
    cudaFree(buffers.queue_a);
    cudaFree(buffers.queue_b);
    cudaFree(buffers.next_flag);
    cudaFree(buffers.count);
    buffers = {};
}

static int run_reconstruction(
    Buffers& buffers,
    int width,
    int height,
    bool complement,
    std::uint8_t* output) {
    constexpr int block = 256;
    const int pixels = width * height;
    const int pixel_grid = (pixels + block - 1) / block;
    const int row_grid = (height + block - 1) / block;
    const int column_grid = (width + block - 1) / block;

    initialize_marker<<<pixel_grid, block>>>(
        buffers.image, buffers.marker, pixels, complement);
    scan_rows_forward<<<row_grid, block>>>(
        buffers.image, buffers.marker, width, height, complement);
    scan_columns_forward<<<column_grid, block>>>(
        buffers.image, buffers.marker, width, height, complement);
    scan_rows_backward<<<row_grid, block>>>(
        buffers.image, buffers.marker, width, height, complement);
    scan_columns_backward<<<column_grid, block>>>(
        buffers.image, buffers.marker, width, height, complement);
    CUDA_CHECK(cudaMemset(buffers.count, 0, sizeof(int)));
    seed_wavefront<<<pixel_grid, block>>>(
        buffers.image, buffers.marker, buffers.queue_a, buffers.count,
        width, height, complement);
    CUDA_CHECK(cudaGetLastError());

    int current_count = 0;
    CUDA_CHECK(cudaMemcpy(
        &current_count, buffers.count, sizeof(int),
        cudaMemcpyDeviceToHost));
    int* current_queue = buffers.queue_a;
    int* next_queue = buffers.queue_b;
    int rounds = 0;
    while (current_count > 0) {
        CUDA_CHECK(cudaMemset(
            buffers.next_flag, 0, pixels * sizeof(int)));
        CUDA_CHECK(cudaMemset(buffers.count, 0, sizeof(int)));
        const int grid = (current_count + block - 1) / block;
        propagate_wavefront<<<grid, block>>>(
            buffers.image, buffers.marker,
            current_queue, current_count,
            next_queue, buffers.count, buffers.next_flag,
            width, height, complement);
        CUDA_CHECK(cudaGetLastError());
        int next_count = 0;
        CUDA_CHECK(cudaMemcpy(
            &next_count, buffers.count, sizeof(int),
            cudaMemcpyDeviceToHost));
        current_count = next_count;
        std::swap(current_queue, next_queue);
        ++rounds;
    }
    materialize_extrema<<<pixel_grid, block>>>(
        buffers.image, buffers.marker, output, pixels, complement);
    CUDA_CHECK(cudaGetLastError());
    return rounds;
}

static int run_task_complete(
    Buffers& buffers,
    int width,
    int height) {
    const int max_rounds = run_reconstruction(
        buffers, width, height, false, buffers.maxima);
    const int min_rounds = run_reconstruction(
        buffers, width, height, true, buffers.minima);
    return max_rounds + min_rounds;
}

static double percentile(std::vector<double> values, double probability) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position =
        probability * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

static Timing benchmark(
    Buffers& buffers,
    int width,
    int height,
    int warmup,
    int repetitions,
    cv::Mat& maxima,
    cv::Mat& minima) {
    const int pixels = width * height;
    for (int i = 0; i < warmup; ++i) {
        run_task_complete(buffers, width, height);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t finish;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&finish));
    std::vector<double> elapsed;
    std::vector<double> rounds;
    elapsed.reserve(static_cast<std::size_t>(repetitions));
    rounds.reserve(static_cast<std::size_t>(repetitions));
    for (int i = 0; i < repetitions; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        const int iteration_rounds =
            run_task_complete(buffers, width, height);
        CUDA_CHECK(cudaEventRecord(finish));
        CUDA_CHECK(cudaEventSynchronize(finish));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, finish));
        elapsed.push_back(milliseconds);
        rounds.push_back(static_cast<double>(iteration_rounds));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(finish));

    maxima.create(height, width, CV_8UC1);
    minima.create(height, width, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(
        maxima.data, buffers.maxima, pixels, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        minima.data, buffers.minima, pixels, cudaMemcpyDeviceToHost));
    return Timing{
        percentile(elapsed, 0.5),
        percentile(elapsed, 0.95),
        percentile(rounds, 0.5),
        percentile(rounds, 0.95)};
}

static bool ends_with_ci(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) {
        return false;
    }
    return std::equal(
        suffix.rbegin(), suffix.rend(), value.rbegin(),
        [](char left, char right) {
            return std::tolower(static_cast<unsigned char>(left)) ==
                   std::tolower(static_cast<unsigned char>(right));
        });
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

static std::uint64_t compare_masks(
    const cv::Mat& first,
    const cv::Mat& second) {
    if (first.rows != second.rows || first.cols != second.cols) {
        throw std::runtime_error("Mask dimensions differ");
    }
    std::uint64_t mismatches = 0;
    for (int y = 0; y < first.rows; ++y) {
        const auto* a = first.ptr<std::uint8_t>(y);
        const auto* b = second.ptr<std::uint8_t>(y);
        for (int x = 0; x < first.cols; ++x) {
            mismatches += a[x] != b[x] ? 1ull : 0ull;
        }
    }
    return mismatches;
}

static std::string quote_csv(const std::string& value) {
    std::string escaped = "\"";
    for (char character : value) {
        if (character == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "Usage: iwpp_reconstruction_cuda_v0_1.exe <image_dir>"
               " --output=DIR [--expected-max=DIR --expected-min=DIR]"
               " [--device=0] [--warmup=1] [--reps=5]\n";
        return 2;
    }
    try {
        const fs::path image_dir = argv[1];
        fs::path output_dir;
        fs::path expected_max_dir;
        fs::path expected_min_dir;
        int device = 0;
        int warmup = 1;
        int repetitions = 5;
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

        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        fs::create_directories(output_dir);
        std::ofstream csv(output_dir / "iwpp_reconstruction_v0_1.csv");
        csv
            << "algorithm,image,width,height,megapixels,warmup,repetitions,"
               "total_allocated_bytes,total_allocated_Bpp,"
               "auxiliary_bytes,auxiliary_Bpp,"
               "median_compute_ms,p95_compute_ms,compute_mpix_s,"
               "median_rounds,p95_rounds,"
               "oracle_max_mismatches,oracle_min_mismatches\n";
        std::ofstream environment(output_dir / "environment.txt");
        environment
            << "harness=iwpp_reconstruction_cuda_v0.1\n"
            << "baseline=Teodoro-style fast-parallel hybrid reconstruction\n"
            << "device_name=" << properties.name << '\n'
            << "connectivity=8\n"
            << "initialization=four_axis_gpu_scans\n"
            << "propagation=round_queues+atomicMax+per_round_dedup\n"
            << "timing=resident_input_preallocated_task_complete\n"
            << "memory=input+two_outputs+int_marker+two_int_queues+int_flag\n"
            << "warmup=" << warmup << '\n'
            << "repetitions=" << repetitions << '\n';

        const auto files = list_images(image_dir);
        std::uint64_t total_mismatches = 0;
        for (const fs::path& image_path : files) {
            cv::Mat image =
                cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
            if (image.empty()) {
                throw std::runtime_error(
                    "Cannot read " + image_path.string());
            }
            if (!image.isContinuous()) {
                image = image.clone();
            }
            const int width = image.cols;
            const int height = image.rows;
            const int pixels = width * height;
            Buffers buffers;
            allocate_buffers(buffers, pixels);
            CUDA_CHECK(cudaMemcpy(
                buffers.image, image.data, pixels, cudaMemcpyHostToDevice));
            cv::Mat maxima;
            cv::Mat minima;
            const Timing timing = benchmark(
                buffers, width, height, warmup, repetitions,
                maxima, minima);

            std::uint64_t max_mismatches = 0;
            std::uint64_t min_mismatches = 0;
            if (!expected_max_dir.empty()) {
                const std::string mask_name =
                    image_path.stem().string() + ".pgm";
                const cv::Mat expected_max = cv::imread(
                    (expected_max_dir / mask_name).string(),
                    cv::IMREAD_GRAYSCALE);
                const cv::Mat expected_min = cv::imread(
                    (expected_min_dir / mask_name).string(),
                    cv::IMREAD_GRAYSCALE);
                if (expected_max.empty() || expected_min.empty()) {
                    throw std::runtime_error(
                        "Missing oracle masks for " +
                        image_path.filename().string());
                }
                max_mismatches = compare_masks(maxima, expected_max);
                min_mismatches = compare_masks(minima, expected_min);
                total_mismatches += max_mismatches + min_mismatches;
            }

            const std::uint64_t total_bytes =
                static_cast<std::uint64_t>(pixels) * 19ull + sizeof(int);
            const std::uint64_t auxiliary_bytes =
                static_cast<std::uint64_t>(pixels) * 16ull + sizeof(int);
            const double megapixels =
                static_cast<double>(pixels) / 1e6;
            const double throughput =
                megapixels / (timing.median_ms / 1000.0);
            csv << quote_csv("IWPP-Reconstruction-CUDA-v0.1") << ','
                << quote_csv(image_path.filename().string()) << ','
                << width << ',' << height << ',' << std::fixed
                << std::setprecision(6) << megapixels << ','
                << warmup << ',' << repetitions << ','
                << total_bytes << ','
                << static_cast<double>(total_bytes) / pixels << ','
                << auxiliary_bytes << ','
                << static_cast<double>(auxiliary_bytes) / pixels << ','
                << timing.median_ms << ',' << timing.p95_ms << ','
                << throughput << ',' << timing.median_rounds << ','
                << timing.p95_rounds << ',' << max_mismatches << ','
                << min_mismatches << '\n';
            csv.flush();
            std::cout
                << "[IWPP-CUDA-v0.1] "
                << image_path.filename().string()
                << " compute_ms=" << timing.median_ms
                << " rounds=" << timing.median_rounds
                << " mismatch=" << max_mismatches + min_mismatches
                << '\n';
            free_buffers(buffers);
        }
        std::cout
            << "[IWPP-CUDA-v0.1-OK] images=" << files.size()
            << " mismatches=" << total_mismatches
            << " output=" << output_dir.string() << '\n';
        return total_mismatches == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 3;
    }
}
