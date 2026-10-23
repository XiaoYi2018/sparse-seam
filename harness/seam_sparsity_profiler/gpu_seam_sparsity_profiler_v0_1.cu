// Sparse-Seam GPU seam-sparsity profiler v0.1
//
// Diagnostic only: this program measures how many tile-local flat-zone
// fragments and pixels would need global seam processing. It is not a
// candidate implementation and its timing is not a paper performance result.

#define NOMINMAX
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
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
        const cudaError_t code = (call);                                        \
        if (code != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(code));                 \
        }                                                                       \
    } while (0)

static bool ends_with_ci(const std::string& value, const std::string& suffix) {
    if (value.size() < suffix.size()) {
        return false;
    }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        const char first = static_cast<char>(std::tolower(
            static_cast<unsigned char>(
                value[value.size() - suffix.size() + i])));
        const char second = static_cast<char>(std::tolower(
            static_cast<unsigned char>(suffix[i])));
        if (first != second) {
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
            ends_with_ci(path, ".jpeg") || ends_with_ci(path, ".bmp") ||
            ends_with_ci(path, ".tif") || ends_with_ci(path, ".tiff") ||
            ends_with_ci(path, ".pgm")) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

static std::vector<int> parse_int_list(const std::string& value) {
    std::vector<int> result;
    std::size_t begin = 0;
    while (begin < value.size()) {
        const std::size_t comma = value.find(',', begin);
        const std::string item = value.substr(
            begin,
            comma == std::string::npos ? std::string::npos : comma - begin);
        result.push_back(std::stoi(item));
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return result;
}

__global__ void quantize_image(
    const std::uint8_t* source,
    std::uint8_t* quantized,
    int count,
    int levels
) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) {
        return;
    }
    if (levels == 256) {
        quantized[index] = source[index];
    } else {
        quantized[index] = static_cast<std::uint8_t>(
            (static_cast<int>(source[index]) * levels) >> 8);
    }
}

__global__ void initialize_parent(int* parent, int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        parent[index] = index;
    }
}

__device__ __forceinline__ int find_root(const int* parent, int node) {
    int root = parent[node];
    while (root != parent[root]) {
        root = parent[root];
    }
    return root;
}

__device__ __forceinline__ void union_min_root(
    int* parent,
    int first,
    int second
) {
    while (true) {
        const int first_root = find_root(parent, first);
        const int second_root = find_root(parent, second);
        if (first_root == second_root) {
            return;
        }
        const int high = max(first_root, second_root);
        const int low = min(first_root, second_root);
        const int old = atomicCAS(parent + high, high, low);
        if (old == high) {
            return;
        }
    }
}

__global__ void union_equal_edges_inside_tiles(
    const std::uint8_t* image,
    int* parent,
    int width,
    int height,
    int tile
) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const int index = y * width + x;
    const std::uint8_t value = image[index];
    const int tile_x = x / tile;
    const int tile_y = y / tile;

    if (x > 0 && (x - 1) / tile == tile_x &&
        image[index - 1] == value) {
        union_min_root(parent, index, index - 1);
    }
    if (y > 0 && (y - 1) / tile == tile_y) {
        if (image[index - width] == value) {
            union_min_root(parent, index, index - width);
        }
        if (x > 0 && (x - 1) / tile == tile_x &&
            image[index - width - 1] == value) {
            union_min_root(parent, index, index - width - 1);
        }
        if (x + 1 < width && (x + 1) / tile == tile_x &&
            image[index - width + 1] == value) {
            union_min_root(parent, index, index - width + 1);
        }
    }
}

__global__ void compress_parent(int* parent, int count) {
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

__global__ void mark_seam_fragments_and_count_edges(
    const std::uint8_t* image,
    const int* parent,
    std::uint32_t* root_touches_seam,
    int width,
    int height,
    int tile,
    unsigned long long* equal_edges,
    unsigned long long* equal_seam_edges
) {
    __shared__ unsigned int block_equal_edges;
    __shared__ unsigned int block_seam_edges;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        block_equal_edges = 0;
        block_seam_edges = 0;
    }
    __syncthreads();

    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    unsigned int local_equal = 0;
    unsigned int local_seam = 0;
    if (x < width && y < height) {
        const int index = y * width + x;
        const std::uint8_t value = image[index];
        const int current_tile_x = x / tile;
        const int current_tile_y = y / tile;
        const int dx[4] = {1, -1, 0, 1};
        const int dy[4] = {0, 1, 1, 1};
        for (int direction = 0; direction < 4; ++direction) {
            const int nx = x + dx[direction];
            const int ny = y + dy[direction];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                continue;
            }
            const int neighbour = ny * width + nx;
            if (image[neighbour] != value) {
                continue;
            }
            ++local_equal;
            const bool crosses_seam =
                nx / tile != current_tile_x || ny / tile != current_tile_y;
            if (crosses_seam) {
                ++local_seam;
                atomicOr(root_touches_seam + parent[index], 1u);
                atomicOr(root_touches_seam + parent[neighbour], 1u);
            }
        }
    }
    if (local_equal != 0) {
        atomicAdd(&block_equal_edges, local_equal);
    }
    if (local_seam != 0) {
        atomicAdd(&block_seam_edges, local_seam);
    }
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(equal_edges, static_cast<unsigned long long>(block_equal_edges));
        atomicAdd(
            equal_seam_edges,
            static_cast<unsigned long long>(block_seam_edges));
    }
}

__global__ void count_fragments_and_pixels(
    const int* parent,
    const std::uint32_t* root_touches_seam,
    int count,
    unsigned long long* total_fragments,
    unsigned long long* seam_fragments,
    unsigned long long* seam_pixels
) {
    __shared__ unsigned int block_total_fragments;
    __shared__ unsigned int block_seam_fragments;
    __shared__ unsigned int block_seam_pixels;
    if (threadIdx.x == 0) {
        block_total_fragments = 0;
        block_seam_fragments = 0;
        block_seam_pixels = 0;
    }
    __syncthreads();

    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        const int root = parent[index];
        if (root == index) {
            atomicAdd(&block_total_fragments, 1u);
            if (root_touches_seam[root] != 0) {
                atomicAdd(&block_seam_fragments, 1u);
            }
        }
        if (root_touches_seam[root] != 0) {
            atomicAdd(&block_seam_pixels, 1u);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(
            total_fragments,
            static_cast<unsigned long long>(block_total_fragments));
        atomicAdd(
            seam_fragments,
            static_cast<unsigned long long>(block_seam_fragments));
        atomicAdd(
            seam_pixels,
            static_cast<unsigned long long>(block_seam_pixels));
    }
}

struct Buffers {
    std::uint8_t* source = nullptr;
    std::uint8_t* quantized = nullptr;
    int* parent = nullptr;
    std::uint32_t* root_touches_seam = nullptr;
    unsigned long long* counters = nullptr;
};

static void allocate_buffers(Buffers& buffers, int count) {
    const std::size_t bytes = static_cast<std::size_t>(count);
    CUDA_CHECK(cudaMalloc(&buffers.source, bytes));
    CUDA_CHECK(cudaMalloc(&buffers.quantized, bytes));
    CUDA_CHECK(cudaMalloc(
        &buffers.parent,
        static_cast<std::size_t>(count) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(
        &buffers.root_touches_seam,
        static_cast<std::size_t>(count) * sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(
        &buffers.counters,
        5 * sizeof(unsigned long long)));
}

static void free_buffers(Buffers& buffers) {
    CUDA_CHECK(cudaFree(buffers.source));
    CUDA_CHECK(cudaFree(buffers.quantized));
    CUDA_CHECK(cudaFree(buffers.parent));
    CUDA_CHECK(cudaFree(buffers.root_touches_seam));
    CUDA_CHECK(cudaFree(buffers.counters));
    buffers = Buffers{};
}

static std::string quote_csv(const std::string& value) {
    std::string result = "\"";
    for (const char character : value) {
        if (character == '"') {
            result += "\"\"";
        } else {
            result += character;
        }
    }
    result += '"';
    return result;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "Usage: gpu_seam_sparsity_profiler_v0_1.exe <image_dir>"
               " --output=FILE [--tiles=32,64,128]"
               " [--levels=256,32,8,2] [--device=0] [--block=16]\n";
        return 2;
    }

    try {
        const fs::path image_dir = argv[1];
        fs::path output_file;
        std::vector<int> tiles{32, 64, 128};
        std::vector<int> levels{256, 32, 8, 2};
        int device = 0;
        int block = 16;
        for (int i = 2; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument.rfind("--output=", 0) == 0) {
                output_file = argument.substr(9);
            } else if (argument.rfind("--tiles=", 0) == 0) {
                tiles = parse_int_list(argument.substr(8));
            } else if (argument.rfind("--levels=", 0) == 0) {
                levels = parse_int_list(argument.substr(9));
            } else if (argument.rfind("--device=", 0) == 0) {
                device = std::stoi(argument.substr(9));
            } else if (argument.rfind("--block=", 0) == 0) {
                block = std::stoi(argument.substr(8));
            }
        }
        if (output_file.empty()) {
            throw std::runtime_error("Missing --output file");
        }
        for (const int tile : tiles) {
            if (tile < 2) {
                throw std::runtime_error("Tile size must be at least 2");
            }
        }
        for (const int level : levels) {
            if (level < 2 || level > 256) {
                throw std::runtime_error("Levels must be in [2,256]");
            }
        }

        CUDA_CHECK(cudaSetDevice(device));
        fs::create_directories(output_file.parent_path());
        std::ofstream csv(output_file);
        if (!csv) {
            throw std::runtime_error("Cannot create output CSV");
        }
        csv << "image,width,height,megapixels,levels,tile,"
               "total_tile_fragments,seam_fragments,interior_fragments,"
               "seam_fragment_fraction,seam_pixels,seam_pixel_fraction,"
               "equal_forward_edges,equal_seam_edges,"
               "seam_edge_fraction,seam_fragments_per_mpix,"
               "equal_seam_edges_per_mpix,diagnostic_ms\n";

        const std::vector<fs::path> files = list_images(image_dir);
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
            const int linear_threads = 256;
            const int linear_blocks =
                (count + linear_threads - 1) / linear_threads;
            const dim3 block_2d(block, block);
            const dim3 grid_2d(
                (width + block - 1) / block,
                (height + block - 1) / block);
            const double megapixels =
                static_cast<double>(count) / 1e6;

            Buffers buffers;
            allocate_buffers(buffers, count);
            CUDA_CHECK(cudaMemcpy(
                buffers.source,
                image.data,
                static_cast<std::size_t>(count),
                cudaMemcpyHostToDevice));

            for (const int level : levels) {
                quantize_image<<<linear_blocks, linear_threads>>>(
                    buffers.source,
                    buffers.quantized,
                    count,
                    level);
                CUDA_CHECK(cudaGetLastError());
                for (const int tile : tiles) {
                    const auto begin =
                        std::chrono::high_resolution_clock::now();
                    initialize_parent<<<linear_blocks, linear_threads>>>(
                        buffers.parent, count);
                    CUDA_CHECK(cudaMemset(
                        buffers.root_touches_seam,
                        0,
                        static_cast<std::size_t>(count) *
                            sizeof(std::uint32_t)));
                    CUDA_CHECK(cudaMemset(
                        buffers.counters,
                        0,
                        5 * sizeof(unsigned long long)));
                    union_equal_edges_inside_tiles<<<grid_2d, block_2d>>>(
                        buffers.quantized,
                        buffers.parent,
                        width,
                        height,
                        tile);
                    compress_parent<<<linear_blocks, linear_threads>>>(
                        buffers.parent, count);
                    mark_seam_fragments_and_count_edges<<<grid_2d, block_2d>>>(
                        buffers.quantized,
                        buffers.parent,
                        buffers.root_touches_seam,
                        width,
                        height,
                        tile,
                        buffers.counters + 3,
                        buffers.counters + 4);
                    count_fragments_and_pixels<<<linear_blocks, linear_threads>>>(
                        buffers.parent,
                        buffers.root_touches_seam,
                        count,
                        buffers.counters,
                        buffers.counters + 1,
                        buffers.counters + 2);
                    CUDA_CHECK(cudaGetLastError());
                    unsigned long long host_counters[5] = {};
                    CUDA_CHECK(cudaMemcpy(
                        host_counters,
                        buffers.counters,
                        sizeof(host_counters),
                        cudaMemcpyDeviceToHost));
                    const auto end =
                        std::chrono::high_resolution_clock::now();

                    const std::uint64_t total_fragments = host_counters[0];
                    const std::uint64_t seam_fragments = host_counters[1];
                    const std::uint64_t seam_pixels = host_counters[2];
                    const std::uint64_t equal_edges = host_counters[3];
                    const std::uint64_t seam_edges = host_counters[4];
                    const std::uint64_t interior_fragments =
                        total_fragments - seam_fragments;
                    const double seam_fragment_fraction =
                        total_fragments == 0
                            ? 0.0
                            : static_cast<double>(seam_fragments) /
                                  static_cast<double>(total_fragments);
                    const double seam_pixel_fraction =
                        static_cast<double>(seam_pixels) /
                        static_cast<double>(count);
                    const double seam_edge_fraction =
                        equal_edges == 0
                            ? 0.0
                            : static_cast<double>(seam_edges) /
                                  static_cast<double>(equal_edges);
                    const double diagnostic_ms =
                        std::chrono::duration<double, std::milli>(
                            end - begin).count();

                    csv << quote_csv(image_path.filename().string()) << ','
                        << width << ',' << height << ',' << std::fixed
                        << std::setprecision(6) << megapixels << ',' << level
                        << ',' << tile << ',' << total_fragments << ','
                        << seam_fragments << ',' << interior_fragments << ','
                        << seam_fragment_fraction << ',' << seam_pixels << ','
                        << seam_pixel_fraction << ',' << equal_edges << ','
                        << seam_edges << ',' << seam_edge_fraction << ','
                        << static_cast<double>(seam_fragments) / megapixels
                        << ',' << static_cast<double>(seam_edges) / megapixels
                        << ',' << diagnostic_ms << '\n';
                    csv.flush();

                    std::cout
                        << "[SEAM-PROFILE] "
                        << image_path.filename().string()
                        << " q=" << level << " tile=" << tile
                        << " fragments=" << total_fragments
                        << " seam_fragments=" << seam_fragments
                        << " seam_fragment_fraction="
                        << seam_fragment_fraction
                        << " seam_pixel_fraction=" << seam_pixel_fraction
                        << " seam_edges=" << seam_edges << '\n';
                }
            }
            free_buffers(buffers);
        }
        std::cout << "[GPU-SEAM-PROFILER-OK] images=" << files.size()
                  << " output=" << output_file.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 3;
    }
}
