// Task-complete external flat-zone baseline for exact 8-neighbour regional
// maxima and minima.
//
// Flat-zone labels are produced by NVIDIA NPP 12.6 LabelMarkersUF. NVIDIA
// documents this routine as based on Chen et al., "An Optimized Union-Find
// Algorithm for Connected Components Labeling Using GPUs." The adapter adds
// only task-required boundary evidence reduction and materialization of both
// extrema masks.

#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_filtering_functions.h>

#include <algorithm>
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

#define NPP_CHECK(call)                                                        \
    do {                                                                       \
        const NppStatus code = (call);                                         \
        if (code != NPP_SUCCESS) {                                             \
            throw std::runtime_error(                                          \
                "NPP error status " + std::to_string(static_cast<int>(code))); \
        }                                                                      \
    } while (0)

struct Buffers {
    Npp8u* image = nullptr;
    Npp32u* labels = nullptr;
    std::uint8_t* has_greater = nullptr;
    std::uint8_t* has_smaller = nullptr;
    std::uint32_t* root_any_greater = nullptr;
    std::uint32_t* root_any_smaller = nullptr;
    std::uint8_t* maxima = nullptr;
    std::uint8_t* minima = nullptr;
    Npp8u* npp_scratch = nullptr;
    int* invalid_label = nullptr;
    int scratch_bytes = 0;
};

struct Timing {
    double median_ms = 0.0;
    double p95_ms = 0.0;
};

__global__ void compute_boundary_flags(
    const std::uint8_t* image,
    std::uint8_t* has_greater,
    std::uint8_t* has_smaller,
    int width,
    int height) {
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

__global__ void reduce_flags_by_npp_label(
    const Npp32u* labels,
    const std::uint8_t* has_greater,
    const std::uint8_t* has_smaller,
    std::uint32_t* root_any_greater,
    std::uint32_t* root_any_smaller,
    int* invalid_label,
    int pixels) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= pixels) {
        return;
    }
    const Npp32u label = labels[index];
    if (label > static_cast<Npp32u>(pixels)) {
        atomicExch(invalid_label, 1);
        return;
    }
    if (has_greater[index] != 0) {
        atomicOr(root_any_greater + label, 1u);
    }
    if (has_smaller[index] != 0) {
        atomicOr(root_any_smaller + label, 1u);
    }
}

__global__ void materialize_masks(
    const Npp32u* labels,
    const std::uint32_t* root_any_greater,
    const std::uint32_t* root_any_smaller,
    std::uint8_t* maxima,
    std::uint8_t* minima,
    int pixels) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= pixels) {
        return;
    }
    const Npp32u label = labels[index];
    maxima[index] = root_any_greater[label] == 0 ? 255 : 0;
    minima[index] = root_any_smaller[label] == 0 ? 255 : 0;
}

static void allocate_buffers(Buffers& buffers, int width, int height) {
    const int pixels = width * height;
    const NppiSize roi{width, height};
    NPP_CHECK(nppiLabelMarkersUFGetBufferSize_32u_C1R(
        roi, &buffers.scratch_bytes));
    if (buffers.scratch_bytes < 1) {
        throw std::runtime_error("NPP returned an invalid scratch size");
    }
    CUDA_CHECK(cudaMalloc(&buffers.image, pixels));
    CUDA_CHECK(cudaMalloc(
        &buffers.labels,
        static_cast<std::size_t>(pixels) * sizeof(Npp32u)));
    CUDA_CHECK(cudaMalloc(&buffers.has_greater, pixels));
    CUDA_CHECK(cudaMalloc(&buffers.has_smaller, pixels));
    CUDA_CHECK(cudaMalloc(
        &buffers.root_any_greater,
        static_cast<std::size_t>(pixels + 1) * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(
        &buffers.root_any_smaller,
        static_cast<std::size_t>(pixels + 1) * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&buffers.maxima, pixels));
    CUDA_CHECK(cudaMalloc(&buffers.minima, pixels));
    CUDA_CHECK(cudaMalloc(&buffers.npp_scratch, buffers.scratch_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.invalid_label, sizeof(int)));
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
    cudaFree(buffers.npp_scratch);
    cudaFree(buffers.invalid_label);
    buffers = Buffers{};
}

static void run_task_complete(
    Buffers& buffers,
    int width,
    int height) {
    const int pixels = width * height;
    const int threads = 256;
    const int blocks = (pixels + threads - 1) / threads;
    const dim3 block_2d(16, 16);
    const dim3 grid_2d(
        (width + block_2d.x - 1) / block_2d.x,
        (height + block_2d.y - 1) / block_2d.y);
    const NppiSize roi{width, height};
    const std::size_t root_bytes =
        static_cast<std::size_t>(pixels + 1) * sizeof(std::uint32_t);

    CUDA_CHECK(cudaMemset(buffers.root_any_greater, 0, root_bytes));
    CUDA_CHECK(cudaMemset(buffers.root_any_smaller, 0, root_bytes));
    CUDA_CHECK(cudaMemset(buffers.invalid_label, 0, sizeof(int)));
    compute_boundary_flags<<<grid_2d, block_2d>>>(
        buffers.image,
        buffers.has_greater,
        buffers.has_smaller,
        width,
        height);
    NPP_CHECK(nppiLabelMarkersUF_8u32u_C1R(
        buffers.image,
        width * static_cast<int>(sizeof(Npp8u)),
        buffers.labels,
        width * static_cast<int>(sizeof(Npp32u)),
        roi,
        nppiNormInf,
        buffers.npp_scratch));
    reduce_flags_by_npp_label<<<blocks, threads>>>(
        buffers.labels,
        buffers.has_greater,
        buffers.has_smaller,
        buffers.root_any_greater,
        buffers.root_any_smaller,
        buffers.invalid_label,
        pixels);
    materialize_masks<<<blocks, threads>>>(
        buffers.labels,
        buffers.root_any_greater,
        buffers.root_any_smaller,
        buffers.maxima,
        buffers.minima,
        pixels);
    CUDA_CHECK(cudaGetLastError());
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

static Timing benchmark(
    Buffers& buffers,
    int width,
    int height,
    int warmup,
    int repetitions,
    cv::Mat& maxima,
    cv::Mat& minima) {
    for (int index = 0; index < warmup; ++index) {
        run_task_complete(buffers, width, height);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t finish = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&finish));
    std::vector<double> elapsed;
    elapsed.reserve(static_cast<std::size_t>(repetitions));
    for (int index = 0; index < repetitions; ++index) {
        CUDA_CHECK(cudaEventRecord(start));
        run_task_complete(buffers, width, height);
        CUDA_CHECK(cudaEventRecord(finish));
        CUDA_CHECK(cudaEventSynchronize(finish));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, finish));
        elapsed.push_back(milliseconds);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(finish));

    int invalid_label = 0;
    CUDA_CHECK(cudaMemcpy(
        &invalid_label,
        buffers.invalid_label,
        sizeof(int),
        cudaMemcpyDeviceToHost));
    if (invalid_label != 0) {
        throw std::runtime_error("NPP emitted a label outside [0, pixels]");
    }

    const int pixels = width * height;
    maxima.create(height, width, CV_8UC1);
    minima.create(height, width, CV_8UC1);
    CUDA_CHECK(cudaMemcpy(
        maxima.data, buffers.maxima, pixels, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        minima.data, buffers.minima, pixels, cudaMemcpyDeviceToHost));
    return Timing{
        percentile(elapsed, 0.5),
        percentile(elapsed, 0.95)};
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
        const auto* first_row = first.ptr<std::uint8_t>(y);
        const auto* second_row = second.ptr<std::uint8_t>(y);
        for (int x = 0; x < first.cols; ++x) {
            mismatches += first_row[x] != second_row[x] ? 1ull : 0ull;
        }
    }
    return mismatches;
}

static std::string quote_csv(const std::string& value) {
    std::string escaped = "\"";
    for (const char character : value) {
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
            << "Usage: npp_labelmarkersuf_extrema_v0_1.exe <image_dir>"
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
        std::ofstream csv(output_dir / "npp_labelmarkersuf_extrema_v0_1.csv");
        csv
            << "algorithm,image,width,height,megapixels,warmup,repetitions,"
               "npp_scratch_bytes,total_allocated_bytes,total_allocated_Bpp,"
               "auxiliary_bytes,auxiliary_Bpp,"
               "median_compute_ms,p95_compute_ms,compute_mpix_s,"
               "oracle_max_mismatches,oracle_min_mismatches\n";
        std::ofstream environment(output_dir / "environment.txt");
        environment
            << "harness=npp_labelmarkersuf_extrema_v0.1\n"
            << "labeling=NVIDIA_NPP_12.6_LabelMarkersUF_8u32u_C1R\n"
            << "documented_basis=Chen_et_al_optimized_GPU_union_find\n"
            << "device_name=" << properties.name << '\n'
            << "connectivity=8\n"
            << "timing=resident_input_preallocated_task_complete\n"
            << "memory=input+labels+NPP_scratch+boundary_flags+"
               "root_flags+two_outputs+error_flag\n"
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
            allocate_buffers(buffers, width, height);
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
                static_cast<std::uint64_t>(pixels) * 17ull +
                static_cast<std::uint64_t>(buffers.scratch_bytes) +
                2ull * sizeof(std::uint32_t) + sizeof(int);
            const std::uint64_t auxiliary_bytes =
                total_bytes - static_cast<std::uint64_t>(pixels) * 3ull;
            const double megapixels = static_cast<double>(pixels) / 1e6;
            const double throughput =
                megapixels / (timing.median_ms / 1000.0);
            csv << quote_csv("NPP-LabelMarkersUF-Extrema-v0.1") << ','
                << quote_csv(image_path.filename().string()) << ','
                << width << ',' << height << ',' << std::fixed
                << std::setprecision(6) << megapixels << ','
                << warmup << ',' << repetitions << ','
                << buffers.scratch_bytes << ',' << total_bytes << ','
                << static_cast<double>(total_bytes) / pixels << ','
                << auxiliary_bytes << ','
                << static_cast<double>(auxiliary_bytes) / pixels << ','
                << timing.median_ms << ',' << timing.p95_ms << ','
                << throughput << ',' << max_mismatches << ','
                << min_mismatches << '\n';
            csv.flush();
            std::cout
                << "[NPP-UF-v0.1] "
                << image_path.filename().string()
                << " compute_ms=" << timing.median_ms
                << " scratch=" << buffers.scratch_bytes
                << " mismatch=" << max_mismatches + min_mismatches
                << '\n';
            free_buffers(buffers);
        }
        std::cout
            << "[NPP-UF-v0.1-OK] images=" << files.size()
            << " mismatches=" << total_mismatches
            << " output=" << output_dir.string() << '\n';
        return total_mismatches == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 3;
    }
}
