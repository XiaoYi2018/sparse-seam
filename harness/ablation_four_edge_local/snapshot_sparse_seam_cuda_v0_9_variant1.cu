// Sparse-Seam: exact dual regional-extrema masks on the GPU.
//
// Exact 8-neighbour grayscale regional maxima/minima:
//   1) build equal-valued flat zones independently inside 64x64 tiles;
//   2) emit only tile-boundary flat-zone fragments to a compact global DSU;
//   3) merge equal fragments across seams and reduce greater/smaller flags;
//   4) rebuild tile-local labels and materialize globally resolved masks.
//
// v0.9 isolates two low-risk classification optimizations:
//   variant 1: division-free cross-tile equality test;
//   variant 2: skip shared atomicOr when local evidence is zero;
//   variant 3: both changes.

#define NOMINMAX
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
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

#ifndef SS_TILE
#define SS_TILE 64
#endif
#ifndef SS_CLASSIFY_VARIANT
#define SS_CLASSIFY_VARIANT 0
#endif
constexpr int TILE = SS_TILE;
constexpr int TILE_PIXELS = TILE * TILE;
constexpr int SLOTS_PER_TILE = 4 * TILE;
constexpr unsigned HAS_GREATER = 1u;
constexpr unsigned HAS_SMALLER = 2u;
constexpr unsigned TOUCHES_SEAM = 4u;
static_assert(TILE == 32 || TILE == 64, "SS_TILE must be 32 or 64");

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t ec = (call);                                          \
        if (ec != cudaSuccess) {                                                \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(ec));                    \
        }                                                                       \
    } while (0)

struct DeviceBuffers {
    std::uint8_t* image = nullptr;
    std::uint8_t* maxima = nullptr;
    std::uint8_t* minima = nullptr;
    std::uint8_t* seam_record = nullptr;
    int* boundary_to_fragment = nullptr;
    int* fragment_parent = nullptr;
    unsigned* fragment_local_bits = nullptr;
    unsigned* fragment_root_bits = nullptr;
    std::uint16_t* tile_fragment_counts = nullptr;
    int* error_flag = nullptr;
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

__device__ __forceinline__ int shared_find_root(const int* parent, int node) {
    int root = parent[node];
    while (root != parent[root]) {
        root = parent[root];
    }
    return root;
}

__device__ __forceinline__ void shared_union_minroot(
    int* parent,
    int first,
    int second
) {
    while (true) {
        const int root_first = shared_find_root(parent, first);
        const int root_second = shared_find_root(parent, second);
        if (root_first == root_second) {
            return;
        }
        const int high = max(root_first, root_second);
        const int low = min(root_first, root_second);
        if (atomicCAS(parent + high, high, low) == high) {
            return;
        }
    }
}

__device__ __forceinline__ int global_find_root(
    const int* parent,
    int node
) {
    int root = parent[node];
    while (root != parent[root]) {
        root = parent[root];
    }
    return root;
}

__device__ __forceinline__ void global_union_minroot(
    int* parent,
    int first,
    int second
) {
    while (true) {
        const int root_first = global_find_root(parent, first);
        const int root_second = global_find_root(parent, second);
        if (root_first == root_second) {
            return;
        }
        const int high = max(root_first, root_second);
        const int low = min(root_first, root_second);
        if (atomicCAS(parent + high, high, low) == high) {
            return;
        }
    }
}

__device__ __forceinline__ int boundary_slot(
    int x,
    int y,
    int width,
    int height,
    int tiles_x
) {
    const int tx = x / TILE;
    const int ty = y / TILE;
    const int start_x = tx * TILE;
    const int start_y = ty * TILE;
    const int tile_width = min(TILE, width - start_x);
    const int tile_height = min(TILE, height - start_y);
    const int lx = x - start_x;
    const int ly = y - start_y;
    const int base = (ty * tiles_x + tx) * SLOTS_PER_TILE;

    if (ly == 0) {
        return base + lx;
    }
    if (ly == tile_height - 1) {
        return base + TILE + lx;
    }
    if (lx == 0) {
        return base + 2 * TILE + (ly - 1);
    }
    if (lx == tile_width - 1) {
        return base + 3 * TILE + (ly - 1);
    }
    return -1;
}

__device__ __forceinline__ void build_tile_flat_zones(
    const std::uint8_t* image,
    int* labels,
    std::uint8_t* tile_image,
    int width,
    int height,
    int start_x,
    int start_y,
    int tile_width,
    int tile_height
) {
    for (int local = static_cast<int>(threadIdx.x);
         local < TILE_PIXELS;
         local += static_cast<int>(blockDim.x)) {
        const int lx = local % TILE;
        const int ly = local / TILE;
        if (lx < tile_width && ly < tile_height) {
            tile_image[local] = image[(start_y + ly) * width + start_x + lx];
            labels[local] = local;
        } else {
            tile_image[local] = 0;
            labels[local] = -1;
        }
    }
    __syncthreads();

    for (int local = static_cast<int>(threadIdx.x);
         local < TILE_PIXELS;
         local += static_cast<int>(blockDim.x)) {
        const int lx = local % TILE;
        const int ly = local / TILE;
        if (lx >= tile_width || ly >= tile_height) {
            continue;
        }
        const std::uint8_t value = tile_image[local];
        if (lx > 0 && tile_image[local - 1] == value) {
            shared_union_minroot(labels, local, local - 1);
        }
        if (ly > 0) {
            if (tile_image[local - TILE] == value) {
                shared_union_minroot(labels, local, local - TILE);
            }
            if (lx > 0 && tile_image[local - TILE - 1] == value) {
                shared_union_minroot(labels, local, local - TILE - 1);
            }
            if (lx + 1 < tile_width &&
                tile_image[local - TILE + 1] == value) {
                shared_union_minroot(labels, local, local - TILE + 1);
            }
        }
    }
    __syncthreads();

    for (int local = static_cast<int>(threadIdx.x);
         local < TILE_PIXELS;
         local += static_cast<int>(blockDim.x)) {
        const int lx = local % TILE;
        const int ly = local / TILE;
        if (lx < tile_width && ly < tile_height) {
            labels[local] = shared_find_root(labels, local);
        }
    }
    __syncthreads();
}

__global__ void classify_tiles_and_emit_fragments(
    const std::uint8_t* image,
    std::uint8_t* maxima,
    std::uint8_t* minima,
    std::uint8_t* seam_record,
    int* boundary_to_fragment,
    int* fragment_parent,
    unsigned* fragment_local_bits,
    unsigned* fragment_root_bits,
    std::uint16_t* tile_fragment_counts,
    int* error_flag,
    int width,
    int height,
    int tiles_x
) {
    __shared__ std::uint8_t tile_image[TILE_PIXELS];
    __shared__ int labels[TILE_PIXELS];
    __shared__ unsigned root_bits[TILE_PIXELS];
    __shared__ int fragment_count;

    const int tx = static_cast<int>(blockIdx.x);
    const int ty = static_cast<int>(blockIdx.y);
    const int start_x = tx * TILE;
    const int start_y = ty * TILE;
    const int tile_width = min(TILE, width - start_x);
    const int tile_height = min(TILE, height - start_y);
    const int tile_id = ty * tiles_x + tx;
    const int fragment_base = tile_id * SLOTS_PER_TILE;

    for (int local = static_cast<int>(threadIdx.x);
         local < TILE_PIXELS;
         local += static_cast<int>(blockDim.x)) {
        root_bits[local] = 0u;
    }
    if (threadIdx.x == 0) {
        fragment_count = 0;
    }
    __syncthreads();

    build_tile_flat_zones(
        image,
        labels,
        tile_image,
        width,
        height,
        start_x,
        start_y,
        tile_width,
        tile_height);

    for (int local = static_cast<int>(threadIdx.x);
         local < TILE_PIXELS;
         local += static_cast<int>(blockDim.x)) {
        const int lx = local % TILE;
        const int ly = local / TILE;
        if (lx >= tile_width || ly >= tile_height) {
            continue;
        }
        const int x = start_x + lx;
        const int y = start_y + ly;
        const std::uint8_t value = tile_image[local];
        unsigned bits = 0u;
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
                bits |= neighbour > value ? HAS_GREATER : 0u;
                bits |= neighbour < value ? HAS_SMALLER : 0u;
                bool crosses_tile = false;
#if SS_CLASSIFY_VARIANT == 1 || SS_CLASSIFY_VARIANT == 3
                crosses_tile =
                    nx < start_x || nx >= start_x + tile_width ||
                    ny < start_y || ny >= start_y + tile_height;
#else
                crosses_tile =
                    nx / TILE != tx || ny / TILE != ty;
#endif
                if (neighbour == value && crosses_tile) {
                    bits |= TOUCHES_SEAM;
                }
            }
        }
#if SS_CLASSIFY_VARIANT == 2 || SS_CLASSIFY_VARIANT == 3
        if (bits != 0u) {
            atomicOr(root_bits + labels[local], bits);
        }
#else
        atomicOr(root_bits + labels[local], bits);
#endif
    }
    __syncthreads();

    for (int local = static_cast<int>(threadIdx.x);
         local < TILE_PIXELS;
         local += static_cast<int>(blockDim.x)) {
        const int lx = local % TILE;
        const int ly = local / TILE;
        if (lx >= tile_width || ly >= tile_height ||
            labels[local] != local ||
            (root_bits[local] & TOUCHES_SEAM) == 0u) {
            continue;
        }
        const int record = atomicAdd(&fragment_count, 1);
        if (record >= 255) {
            atomicExch(error_flag, 1);
        } else {
            root_bits[local] =
                (root_bits[local] & 7u) |
                (static_cast<unsigned>(record + 1) << 3);
        }
    }
    __syncthreads();

    for (int local = static_cast<int>(threadIdx.x);
         local < TILE_PIXELS;
         local += static_cast<int>(blockDim.x)) {
        const int lx = local % TILE;
        const int ly = local / TILE;
        if (lx >= tile_width || ly >= tile_height) {
            continue;
        }
        const int root = labels[local];
        const unsigned packed = root_bits[root];
        if ((packed & TOUCHES_SEAM) == 0u) {
            const int index = (start_y + ly) * width + start_x + lx;
            seam_record[index] = 255u;
            maxima[index] = (packed & HAS_GREATER) == 0u ? 255 : 0;
            minima[index] = (packed & HAS_SMALLER) == 0u ? 255 : 0;
            continue;
        }
        const int record = static_cast<int>(packed >> 3) - 1;
        if (record < 0 || record >= SLOTS_PER_TILE) {
            atomicExch(error_flag, 2);
            continue;
        }
        const int fragment = fragment_base + record;
        const int index = (start_y + ly) * width + start_x + lx;
        seam_record[index] = static_cast<std::uint8_t>(record);
        if (local == root) {
            fragment_parent[fragment] = fragment;
            fragment_local_bits[fragment] = packed & 3u;
            fragment_root_bits[fragment] = 0u;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        tile_fragment_counts[tile_id] =
            static_cast<std::uint16_t>(fragment_count);
    }

    for (int local = static_cast<int>(threadIdx.x);
         local < TILE_PIXELS;
         local += static_cast<int>(blockDim.x)) {
        const int lx = local % TILE;
        const int ly = local / TILE;
        if (lx >= tile_width || ly >= tile_height) {
            continue;
        }
        const unsigned packed = root_bits[labels[local]];
        if ((packed & TOUCHES_SEAM) == 0u) {
            continue;
        }
        const int slot = boundary_slot(
            start_x + lx,
            start_y + ly,
            width,
            height,
            tiles_x);
        if (slot >= 0) {
            const int record = static_cast<int>(packed >> 3) - 1;
            boundary_to_fragment[slot] = fragment_base + record;
        }
    }
}

__global__ void materialize_saved_seam_records(
    std::uint8_t* maxima,
    std::uint8_t* minima,
    const std::uint8_t* seam_record,
    const int* fragment_parent,
    const unsigned* fragment_root_bits,
    int width,
    int height,
    int tiles_x
) {
    const int index =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = width * height;
    if (index >= count) {
        return;
    }
    const std::uint8_t record = seam_record[index];
    if (record == 255u) {
        return;
    }
    const int x = index % width;
    const int y = index / width;
    const int tile_id = (y / TILE) * tiles_x + (x / TILE);
    const int fragment =
        tile_id * SLOTS_PER_TILE + static_cast<int>(record);
    const int root = fragment_parent[fragment];
    const unsigned bits = fragment_root_bits[root];
    maxima[index] = (bits & HAS_GREATER) == 0u ? 255 : 0;
    minima[index] = (bits & HAS_SMALLER) == 0u ? 255 : 0;
}

__device__ __forceinline__ void merge_boundary_pair_if_equal(
    const std::uint8_t* image,
    const int* boundary_to_fragment,
    int* fragment_parent,
    int x,
    int y,
    int nx,
    int ny,
    int width,
    int height,
    int tiles_x
) {
    if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
        return;
    }
    if (image[y * width + x] != image[ny * width + nx]) {
        return;
    }
    const int first_slot = boundary_slot(x, y, width, height, tiles_x);
    const int second_slot = boundary_slot(nx, ny, width, height, tiles_x);
    if (first_slot < 0 || second_slot < 0) {
        return;
    }
    const int first = boundary_to_fragment[first_slot];
    const int second = boundary_to_fragment[second_slot];
    if (first >= 0 && second >= 0) {
        global_union_minroot(fragment_parent, first, second);
    }
}

__global__ void merge_equal_fragments_on_compact_seams(
    const std::uint8_t* image,
    const int* boundary_to_fragment,
    int* fragment_parent,
    int width,
    int height,
    int tiles_x,
    int tiles_y
) {
    const int entry =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int vertical_entries = max(0, tiles_x - 1) * height;
    const int horizontal_entries = max(0, tiles_y - 1) * width;
    if (entry < vertical_entries) {
        const int seam = entry / height + 1;
        const int y = entry % height;
        const int x = seam * TILE - 1;
        merge_boundary_pair_if_equal(
            image, boundary_to_fragment, fragment_parent,
            x, y, x + 1, y - 1, width, height, tiles_x);
        merge_boundary_pair_if_equal(
            image, boundary_to_fragment, fragment_parent,
            x, y, x + 1, y, width, height, tiles_x);
        merge_boundary_pair_if_equal(
            image, boundary_to_fragment, fragment_parent,
            x, y, x + 1, y + 1, width, height, tiles_x);
        return;
    }
    const int horizontal_entry = entry - vertical_entries;
    if (horizontal_entry < horizontal_entries) {
        const int seam = horizontal_entry / width + 1;
        const int x = horizontal_entry % width;
        const int y = seam * TILE - 1;
        merge_boundary_pair_if_equal(
            image, boundary_to_fragment, fragment_parent,
            x, y, x - 1, y + 1, width, height, tiles_x);
        merge_boundary_pair_if_equal(
            image, boundary_to_fragment, fragment_parent,
            x, y, x, y + 1, width, height, tiles_x);
        merge_boundary_pair_if_equal(
            image, boundary_to_fragment, fragment_parent,
            x, y, x + 1, y + 1, width, height, tiles_x);
    }
}

__global__ void compress_fragment_parents(
    int* parent,
    const std::uint16_t* tile_fragment_counts,
    int count
) {
    const int index =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) {
        return;
    }
    const int tile = index / SLOTS_PER_TILE;
    const int record = index - tile * SLOTS_PER_TILE;
    if (record >= static_cast<int>(tile_fragment_counts[tile])) {
        return;
    }
    parent[index] = global_find_root(parent, index);
}

__global__ void reduce_fragment_bits(
    const int* parent,
    const unsigned* local_bits,
    unsigned* root_bits,
    const std::uint16_t* tile_fragment_counts,
    int count
) {
    const int index =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= count) {
        return;
    }
    const int tile = index / SLOTS_PER_TILE;
    const int record = index - tile * SLOTS_PER_TILE;
    if (record >= static_cast<int>(tile_fragment_counts[tile])) {
        return;
    }
    atomicOr(root_bits + parent[index], local_bits[index] & 3u);
}

__global__ void materialize_seam_components(
    const std::uint8_t* image,
    std::uint8_t* maxima,
    std::uint8_t* minima,
    const int* boundary_to_fragment,
    const int* fragment_parent,
    const unsigned* fragment_root_bits,
    int* error_flag,
    int width,
    int height,
    int tiles_x
) {
    __shared__ std::uint8_t tile_image[TILE_PIXELS];
    __shared__ int labels[TILE_PIXELS];
    __shared__ int root_fragment[TILE_PIXELS];

    const int tx = static_cast<int>(blockIdx.x);
    const int ty = static_cast<int>(blockIdx.y);
    const int start_x = tx * TILE;
    const int start_y = ty * TILE;
    const int tile_width = min(TILE, width - start_x);
    const int tile_height = min(TILE, height - start_y);

    for (int local = static_cast<int>(threadIdx.x);
         local < TILE_PIXELS;
         local += static_cast<int>(blockDim.x)) {
        root_fragment[local] = -1;
    }
    __syncthreads();

    build_tile_flat_zones(
        image,
        labels,
        tile_image,
        width,
        height,
        start_x,
        start_y,
        tile_width,
        tile_height);

    for (int local = static_cast<int>(threadIdx.x);
         local < TILE_PIXELS;
         local += static_cast<int>(blockDim.x)) {
        const int lx = local % TILE;
        const int ly = local / TILE;
        if (lx >= tile_width || ly >= tile_height) {
            continue;
        }
        const int slot = boundary_slot(
            start_x + lx,
            start_y + ly,
            width,
            height,
            tiles_x);
        if (slot < 0) {
            continue;
        }
        const int fragment = boundary_to_fragment[slot];
        if (fragment >= 0) {
            atomicCAS(root_fragment + labels[local], -1, fragment);
        }
    }
    __syncthreads();

    for (int local = static_cast<int>(threadIdx.x);
         local < TILE_PIXELS;
         local += static_cast<int>(blockDim.x)) {
        const int lx = local % TILE;
        const int ly = local / TILE;
        if (lx >= tile_width || ly >= tile_height) {
            continue;
        }
        const int fragment = root_fragment[labels[local]];
        if (fragment < 0) {
            continue;
        }
        const int root = fragment_parent[fragment];
        if (root < 0) {
            atomicExch(error_flag, 3);
            continue;
        }
        const unsigned bits = fragment_root_bits[root];
        const int index = (start_y + ly) * width + start_x + lx;
        maxima[index] = (bits & HAS_GREATER) == 0u ? 255 : 0;
        minima[index] = (bits & HAS_SMALLER) == 0u ? 255 : 0;
    }
}

static void allocate_buffers(
    DeviceBuffers& buffers,
    int pixels,
    int fragment_capacity
) {
    const std::size_t pixel_bytes =
        static_cast<std::size_t>(pixels) * sizeof(std::uint8_t);
    const std::size_t fragment_int_bytes =
        static_cast<std::size_t>(fragment_capacity) * sizeof(int);
    const std::size_t fragment_uint_bytes =
        static_cast<std::size_t>(fragment_capacity) * sizeof(unsigned);
    CUDA_CHECK(cudaMalloc(&buffers.image, pixel_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.maxima, pixel_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.minima, pixel_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.seam_record, pixel_bytes));
    CUDA_CHECK(cudaMalloc(
        &buffers.boundary_to_fragment, fragment_int_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.fragment_parent, fragment_int_bytes));
    CUDA_CHECK(cudaMalloc(
        &buffers.fragment_local_bits, fragment_uint_bytes));
    CUDA_CHECK(cudaMalloc(
        &buffers.fragment_root_bits, fragment_uint_bytes));
    CUDA_CHECK(cudaMalloc(
        &buffers.tile_fragment_counts,
        static_cast<std::size_t>(fragment_capacity / SLOTS_PER_TILE) *
            sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMalloc(&buffers.error_flag, sizeof(int)));
}

static void free_buffers(DeviceBuffers& buffers) {
    CUDA_CHECK(cudaFree(buffers.image));
    CUDA_CHECK(cudaFree(buffers.maxima));
    CUDA_CHECK(cudaFree(buffers.minima));
    CUDA_CHECK(cudaFree(buffers.seam_record));
    CUDA_CHECK(cudaFree(buffers.boundary_to_fragment));
    CUDA_CHECK(cudaFree(buffers.fragment_parent));
    CUDA_CHECK(cudaFree(buffers.fragment_local_bits));
    CUDA_CHECK(cudaFree(buffers.fragment_root_bits));
    CUDA_CHECK(cudaFree(buffers.tile_fragment_counts));
    CUDA_CHECK(cudaFree(buffers.error_flag));
    buffers = DeviceBuffers{};
}

static float run_sparse_seam(
    DeviceBuffers& buffers,
    const cv::Mat& image,
    cv::Mat& maxima,
    cv::Mat& minima,
    int& active_fragments
) {
    const int width = image.cols;
    const int height = image.rows;
    const int pixels = width * height;
    const int tiles_x = (width + TILE - 1) / TILE;
    const int tiles_y = (height + TILE - 1) / TILE;
    const int fragment_capacity = tiles_x * tiles_y * SLOTS_PER_TILE;
    const std::size_t pixel_bytes =
        static_cast<std::size_t>(pixels) * sizeof(std::uint8_t);
    CUDA_CHECK(cudaMemcpy(
        buffers.image, image.data, pixel_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(buffers.error_flag, 0, sizeof(int)));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    const dim3 tile_grid(tiles_x, tiles_y);
    constexpr int threads = 256;
    const int pixel_blocks = (pixels + threads - 1) / threads;
    const int seam_entries =
        std::max(0, tiles_x - 1) * height +
        std::max(0, tiles_y - 1) * width;
    const int seam_blocks = (seam_entries + threads - 1) / threads;
    const int fragment_blocks =
        (fragment_capacity + threads - 1) / threads;

    classify_tiles_and_emit_fragments<<<tile_grid, threads>>>(
        buffers.image,
        buffers.maxima,
        buffers.minima,
        buffers.seam_record,
        buffers.boundary_to_fragment,
        buffers.fragment_parent,
        buffers.fragment_local_bits,
        buffers.fragment_root_bits,
        buffers.tile_fragment_counts,
        buffers.error_flag,
        width,
        height,
        tiles_x);
    if (seam_blocks > 0) {
        merge_equal_fragments_on_compact_seams<<<seam_blocks, threads>>>(
            buffers.image,
            buffers.boundary_to_fragment,
            buffers.fragment_parent,
            width,
            height,
            tiles_x,
            tiles_y);
    }
    compress_fragment_parents<<<fragment_blocks, threads>>>(
        buffers.fragment_parent,
        buffers.tile_fragment_counts,
        fragment_capacity);
    reduce_fragment_bits<<<fragment_blocks, threads>>>(
        buffers.fragment_parent,
        buffers.fragment_local_bits,
        buffers.fragment_root_bits,
        buffers.tile_fragment_counts,
        fragment_capacity);
    materialize_saved_seam_records<<<pixel_blocks, threads>>>(
        buffers.maxima,
        buffers.minima,
        buffers.seam_record,
        buffers.fragment_parent,
        buffers.fragment_root_bits,
        width,
        height,
        tiles_x);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    int error_flag = 0;
    CUDA_CHECK(cudaMemcpy(
        &error_flag,
        buffers.error_flag,
        sizeof(int),
        cudaMemcpyDeviceToHost));
    if (error_flag != 0) {
        throw std::runtime_error(
            "Sparse-seam device invariant failed, flag=" +
            std::to_string(error_flag));
    }

    const int tile_count = tiles_x * tiles_y;
    std::vector<std::uint16_t> fragment_counts(
        static_cast<std::size_t>(tile_count), 0);
    CUDA_CHECK(cudaMemcpy(
        fragment_counts.data(),
        buffers.tile_fragment_counts,
        static_cast<std::size_t>(tile_count) * sizeof(std::uint16_t),
        cudaMemcpyDeviceToHost));
    active_fragments = 0;
    for (const std::uint16_t count : fragment_counts) {
        active_fragments += static_cast<int>(count);
    }

    maxima.create(height, width, CV_8U);
    minima.create(height, width, CV_8U);
    CUDA_CHECK(cudaMemcpy(
        maxima.data,
        buffers.maxima,
        pixel_bytes,
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        minima.data,
        buffers.minima,
        pixel_bytes,
        cudaMemcpyDeviceToHost));
    return elapsed_ms;
}

static std::uint64_t compare_masks(
    const cv::Mat& actual,
    const cv::Mat& expected
) {
    if (actual.rows != expected.rows || actual.cols != expected.cols) {
        throw std::runtime_error("Expected/actual mask dimensions differ");
    }
    std::uint64_t mismatches = 0;
    const int count = actual.rows * actual.cols;
    for (int i = 0; i < count; ++i) {
        if ((actual.data[i] != 0) != (expected.data[i] != 0)) {
            ++mismatches;
        }
    }
    return mismatches;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr
            << "Usage: sparse_seam_cuda_v0_9.exe <image_dir>"
               " <expected_max_dir> <expected_min_dir> --output=DIR"
               " [--device=0] [--skip-expected]\n";
        return 2;
    }
    try {
        const fs::path image_dir = argv[1];
        const fs::path expected_max_dir = argv[2];
        const fs::path expected_min_dir = argv[3];
        fs::path output_dir;
        int device = 0;
        bool skip_expected = false;
        for (int i = 4; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument.rfind("--output=", 0) == 0) {
                output_dir = argument.substr(9);
            } else if (argument.rfind("--device=", 0) == 0) {
                device = std::stoi(argument.substr(9));
            } else if (argument == "--skip-expected") {
                skip_expected = true;
            }
        }
        if (output_dir.empty()) {
            throw std::runtime_error("Missing --output directory");
        }

        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        fs::create_directories(output_dir / "actual_max");
        fs::create_directories(output_dir / "actual_min");
        std::ofstream csv(output_dir / "validation.csv");
        csv << "image,width,height,tiles,fragment_capacity,active_fragments,"
               "auxiliary_bytes,auxiliary_bytes_per_pixel,elapsed_ms,"
               "oracle_checked,max_mismatches,min_mismatches\n";

        const std::vector<fs::path> files = list_images(image_dir);
        std::uint64_t total_max_mismatches = 0;
        std::uint64_t total_min_mismatches = 0;
        std::uint64_t total_pixels = 0;
        for (const fs::path& image_path : files) {
            cv::Mat image =
                cv::imread(image_path.string(), cv::IMREAD_GRAYSCALE);
            if (image.empty()) {
                throw std::runtime_error(
                    "Cannot read input " + image_path.string());
            }
            if (!image.isContinuous()) {
                image = image.clone();
            }
            cv::Mat expected_max;
            cv::Mat expected_min;
            if (!skip_expected) {
                const fs::path expected_name =
                    expected_max_dir / (image_path.stem().string() + ".pgm");
                const fs::path expected_min_name =
                    expected_min_dir / (image_path.stem().string() + ".pgm");
                expected_max =
                    cv::imread(expected_name.string(), cv::IMREAD_GRAYSCALE);
                expected_min =
                    cv::imread(expected_min_name.string(), cv::IMREAD_GRAYSCALE);
                if (expected_max.empty() || expected_min.empty()) {
                    throw std::runtime_error(
                        "Missing expected masks for " +
                        image_path.filename().string());
                }
                if (!expected_max.isContinuous()) {
                    expected_max = expected_max.clone();
                }
                if (!expected_min.isContinuous()) {
                    expected_min = expected_min.clone();
                }
            }

            const int width = image.cols;
            const int height = image.rows;
            const int pixels = width * height;
            const int tiles_x = (width + TILE - 1) / TILE;
            const int tiles_y = (height + TILE - 1) / TILE;
            const int tiles = tiles_x * tiles_y;
            const int fragment_capacity = tiles * SLOTS_PER_TILE;
            DeviceBuffers buffers;
            allocate_buffers(buffers, pixels, fragment_capacity);
            cv::Mat actual_max;
            cv::Mat actual_min;
            int active_fragments = 0;
            const float elapsed_ms = run_sparse_seam(
                buffers,
                image,
                actual_max,
                actual_min,
                active_fragments);
            const std::uint64_t max_mismatches =
                skip_expected ? 0 : compare_masks(actual_max, expected_max);
            const std::uint64_t min_mismatches =
                skip_expected ? 0 : compare_masks(actual_min, expected_min);
            total_max_mismatches += max_mismatches;
            total_min_mismatches += min_mismatches;
            total_pixels += static_cast<std::uint64_t>(pixels);

            const std::uint64_t auxiliary_bytes =
                static_cast<std::uint64_t>(pixels) +
                static_cast<std::uint64_t>(fragment_capacity) * 16ull +
                static_cast<std::uint64_t>(tiles) *
                    sizeof(std::uint16_t);
            csv << '"' << image_path.filename().string() << "\","
                << width << ',' << height << ',' << tiles << ','
                << fragment_capacity << ',' << active_fragments << ','
                << auxiliary_bytes << ',' << std::fixed
                << std::setprecision(6)
                << static_cast<double>(auxiliary_bytes) /
                       static_cast<double>(pixels)
                << ',' << elapsed_ms << ',' << (skip_expected ? 0 : 1)
                << ',' << max_mismatches << ','
                << min_mismatches << '\n';
            csv.flush();

            if (!skip_expected) {
                cv::imwrite(
                    (output_dir / "actual_max" /
                     (image_path.stem().string() + ".pgm")).string(),
                    actual_max);
                cv::imwrite(
                    (output_dir / "actual_min" /
                     (image_path.stem().string() + ".pgm")).string(),
                    actual_min);
            }
            std::cout
                << "[SPARSE-SEAM] " << image_path.filename().string()
                << ' ' << width << 'x' << height
                << " active_fragments=" << active_fragments
                << " aux_Bpp="
                << static_cast<double>(auxiliary_bytes) /
                       static_cast<double>(pixels)
                << " elapsed_ms=" << elapsed_ms
                << " max_mismatch=" << max_mismatches
                << " min_mismatch=" << min_mismatches << '\n';
            free_buffers(buffers);
        }

        std::ofstream environment(output_dir / "environment.txt");
        environment
            << "algorithm=Sparse-Seam-CUDA-v0.9\n"
            << "device_name=" << properties.name << '\n'
            << "compute_capability=" << properties.major << '.'
            << properties.minor << '\n'
            << "tile=" << TILE << '\n'
            << "threads_per_tile=256\n"
            << "fragment_slots_per_tile=" << SLOTS_PER_TILE << '\n'
            << "auxiliary_state_definition=seam_record_per_pixel+"
               "boundary_map+fragment_parent+fragment_local_bits+"
               "fragment_root_bits+tile_fragment_counts\n"
            << "initialization=fused_into_tile_classification\n"
            << "classification_variant=" << SS_CLASSIFY_VARIANT << '\n'
            << "asymptotic_auxiliary_bytes_per_full_tile_pixel="
            << (1.0 + 64.0 / static_cast<double>(TILE) +
                2.0 / static_cast<double>(TILE * TILE))
            << '\n';
        environment
            << "oracle_checked=" << (skip_expected ? 0 : 1) << '\n';
        std::cout
            << (skip_expected ? "[SPARSE-SEAM-MEASUREMENT-OK"
                              : "[SPARSE-SEAM-VALIDATION-OK")
            << "] images=" << files.size()
            << " pixels=" << total_pixels
            << " max_mismatches=" << total_max_mismatches
            << " min_mismatches=" << total_min_mismatches
            << " output=" << output_dir.string() << '\n';
        return skip_expected ||
                       (total_max_mismatches == 0 &&
                        total_min_mismatches == 0)
                   ? 0
                   : 1;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 3;
    }
}
