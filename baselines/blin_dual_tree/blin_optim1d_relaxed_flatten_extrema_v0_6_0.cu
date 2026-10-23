// Independent task-complete reproduction of the hierarchical CUDA Base
// pipeline described by Blin et al., IEEE TPDS 2022.
//
// This is not the authors' source. It independently maps the paper's Optim-1D
// organization and 8-neighbor grid simplification to 16x16 shared-memory
// tiles, applies the same simplification to tile boundaries with a conservative
// 2x2 tile-corner repair, then performs a post-merge non-atomic canonical
// flatten and direct canonical-parent max/min mask extraction.

#define BLIN_V01_COMMON_ONLY
#include "blin_sortless_maxtree_extrema_v0_1_common_snapshot.cuh"
#undef BLIN_V01_COMMON_ONLY

constexpr int kTile = 16;
constexpr int kTilePixels = kTile * kTile;

__global__ void build_local_trees_tiled(
    const std::uint8_t* image,
    int* global_parent,
    int width,
    int height,
    bool complement,
    int* global_invalid) {
    __shared__ std::uint8_t levels[kTilePixels];
    __shared__ int parent[kTilePixels];
    __shared__ int local_invalid;

    const int local_x = static_cast<int>(threadIdx.x);
    const int local_y = static_cast<int>(threadIdx.y);
    const int local = local_y * kTile + local_x;
    const int x = static_cast<int>(blockIdx.x) * kTile + local_x;
    const int y = static_cast<int>(blockIdx.y) * kTile + local_y;
    const bool valid = x < width && y < height;
    const int global = y * width + x;

    if (local == 0) {
        local_invalid = 0;
    }
    levels[local] = valid ? image[global] : 0;
    parent[local] = -1;
    __syncthreads();

    if (valid) {
        if (local_x + 1 < kTile && x + 1 < width) {
            connect_trees(
                levels, parent, local, local + 1, complement,
                kTilePixels, &local_invalid);
        }
        if (local_y + 1 < kTile && y + 1 < height) {
            if (local_x > 0) {
                connect_trees(
                    levels, parent, local, local + kTile - 1, complement,
                    kTilePixels, &local_invalid);
            }
            connect_trees(
                levels, parent, local, local + kTile, complement,
                kTilePixels, &local_invalid);
            if (local_x + 1 < kTile && x + 1 < width) {
                connect_trees(
                    levels, parent, local, local + kTile + 1, complement,
                    kTilePixels, &local_invalid);
            }
        }
    }
    __syncthreads();

    if (valid) {
        const int current_parent = atomic_read_parent(parent + local);
        if (current_parent != -1) {
            const TreePosition canonical = find_level_root(
                levels,
                parent,
                current_parent,
                complement,
                kTilePixels,
                &local_invalid);
            parent[local] = canonical.node;
        }
    }
    __syncthreads();

    if (valid) {
        const int local_parent = parent[local];
        if (local_parent == -1) {
            global_parent[global] = -1;
        } else {
            const int parent_x =
                static_cast<int>(blockIdx.x) * kTile +
                local_parent % kTile;
            const int parent_y =
                static_cast<int>(blockIdx.y) * kTile +
                local_parent / kTile;
            global_parent[global] = parent_y * width + parent_x;
        }
    }
    if (local == 0 && local_invalid != 0) {
        atomicExch(global_invalid, 1);
    }
}

__global__ void build_local_trees_optim1d(
    const std::uint8_t* image,
    int* global_parent,
    int width,
    int height,
    bool complement,
    int* global_invalid) {
    __shared__ std::uint8_t levels[kTilePixels];
    __shared__ int parent[kTilePixels];
    __shared__ int stacks[kTilePixels];
    __shared__ int local_invalid;

    const int column = static_cast<int>(threadIdx.x);
    const int tile_x = static_cast<int>(blockIdx.x) * kTile;
    const int tile_y = static_cast<int>(blockIdx.y) * kTile;
    const int x = tile_x + column;
    const int valid_height = min(kTile, height - tile_y);
    const bool valid_column = column < kTile && x < width;

    if (column == 0) {
        local_invalid = 0;
    }
    for (int row = 0; row < kTile; ++row) {
        const int local = row * kTile + column;
        const int y = tile_y + row;
        levels[local] =
            valid_column && y < height ? image[y * width + x] : 0;
        parent[local] = -1;
    }
    __syncthreads();

    if (valid_column && valid_height > 0) {
        int depth = 0;
        int root = (valid_height - 1) * kTile + column;
        for (int row = valid_height - 2; row >= 0; --row) {
            const int pixel = row * kTile + column;
            const int root_level =
                tree_level(levels, root, complement);
            const int pixel_level =
                tree_level(levels, pixel, complement);
            if (root_level < pixel_level) {
                stacks[column * kTile + depth++] = root;
                root = pixel;
            } else if (root_level == pixel_level) {
                parent[pixel] = root;
            } else {
                while (depth > 0 &&
                       pixel_level <= tree_level(
                           levels,
                           stacks[column * kTile + depth - 1],
                           complement)) {
                    parent[root] =
                        stacks[column * kTile + --depth];
                    root = parent[root];
                }
                if (tree_level(levels, root, complement) > pixel_level) {
                    parent[root] = pixel;
                    root = pixel;
                } else {
                    parent[pixel] = root;
                }
            }
        }
        while (depth > 0) {
            parent[root] = stacks[column * kTile + --depth];
            root = parent[root];
        }
        parent[root] = -1;
    }
    __syncthreads();

    if (valid_column && column + 1 < kTile && x + 1 < width) {
        for (int row = 0; row < valid_height; ++row) {
            const int left = row * kTile + column;
            const int right = left + 1;
            int first = -1;
            int second = -1;
            bool horizontal = false;
            if (row == 0) {
                first = left;
                second = right;
                horizontal = true;
            } else {
                const int upper_left = left - kTile;
                const int upper_right = right - kTile;
                const int left_level =
                    tree_level(levels, left, complement);
                const int right_level =
                    tree_level(levels, right, complement);
                const int upper_left_level =
                    tree_level(levels, upper_left, complement);
                const int upper_right_level =
                    tree_level(levels, upper_right, complement);
                if (left_level > upper_left_level &&
                    right_level > upper_right_level) {
                    first = left;
                    second = right;
                    horizontal = true;
                } else if (left_level <= upper_left_level &&
                           right_level > upper_right_level) {
                    first = upper_left;
                    second = right;
                } else if (left_level > upper_left_level &&
                           right_level <= upper_right_level) {
                    first = left;
                    second = upper_right;
                }
            }
            if (horizontal && row + 1 < valid_height) {
                const int lower_left = left + kTile;
                const int lower_right = right + kTile;
                if (tree_level(levels, left, complement) <
                        tree_level(levels, lower_left, complement) ||
                    tree_level(levels, right, complement) <
                        tree_level(levels, lower_right, complement)) {
                    first = -1;
                    second = -1;
                }
            }
            if (first != -1) {
                connect_trees(
                    levels, parent, first, second, complement,
                    kTilePixels, &local_invalid);
            }
        }
    }
    __syncthreads();

    if (valid_column) {
        for (int row = 0; row < valid_height; ++row) {
            const int local = row * kTile + column;
            const int current_parent =
                atomic_read_parent(parent + local);
            if (current_parent != -1) {
                const TreePosition canonical = find_level_root(
                    levels,
                    parent,
                    current_parent,
                    complement,
                    kTilePixels,
                    &local_invalid);
                parent[local] = canonical.node;
            }
        }
    }
    __syncthreads();

    if (valid_column) {
        for (int row = 0; row < valid_height; ++row) {
            const int local = row * kTile + column;
            const int global = (tile_y + row) * width + x;
            const int local_parent = parent[local];
            if (local_parent == -1) {
                global_parent[global] = -1;
            } else {
                const int parent_x =
                    tile_x + local_parent % kTile;
                const int parent_y =
                    tile_y + local_parent / kTile;
                global_parent[global] =
                    parent_y * width + parent_x;
            }
        }
    }
    if (column == 0 && local_invalid != 0) {
        atomicExch(global_invalid, 1);
    }
}

__device__ __forceinline__ void merge_simplified_pair(
    const std::uint8_t* image,
    int* parent,
    int current_first,
    int current_second,
    int previous_first,
    int previous_second,
    int next_first,
    int next_second,
    bool has_previous,
    bool has_next,
    bool complement,
    int pixels,
    int* invalid_tree) {
    int first = -1;
    int second = -1;
    bool direct = false;
    if (!has_previous) {
        first = current_first;
        second = current_second;
        direct = true;
    } else {
        const int first_level =
            tree_level(image, current_first, complement);
        const int second_level =
            tree_level(image, current_second, complement);
        const int previous_first_level =
            tree_level(image, previous_first, complement);
        const int previous_second_level =
            tree_level(image, previous_second, complement);
        if (first_level > previous_first_level &&
            second_level > previous_second_level) {
            first = current_first;
            second = current_second;
            direct = true;
        } else if (first_level <= previous_first_level &&
                   second_level > previous_second_level) {
            first = previous_first;
            second = current_second;
        } else if (first_level > previous_first_level &&
                   second_level <= previous_second_level) {
            first = current_first;
            second = previous_second;
        }
    }
    if (direct && has_next) {
        if (tree_level(image, current_first, complement) <
                tree_level(image, next_first, complement) ||
            tree_level(image, current_second, complement) <
                tree_level(image, next_second, complement)) {
            first = -1;
            second = -1;
        }
    }
    if (first != -1) {
        connect_trees(
            image, parent, first, second,
            complement, pixels, invalid_tree);
    }
}

__global__ void merge_vertical_tile_boundaries_simplified(
    const std::uint8_t* image,
    int* parent,
    int width,
    int height,
    int vertical_boundaries,
    bool complement,
    int* invalid_tree) {
    const int task =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int tasks = vertical_boundaries * height;
    if (task >= tasks) {
        return;
    }
    const int boundary = task / height;
    const int y = task - boundary * height;
    const int right_x = (boundary + 1) * kTile;
    const int left = y * width + right_x - 1;
    const int right = left + 1;
    const bool has_previous = y > 0;
    const bool has_next = y + 1 < height;
    merge_simplified_pair(
        image,
        parent,
        left,
        right,
        left - width,
        right - width,
        left + width,
        right + width,
        has_previous,
        has_next,
        complement,
        width * height,
        invalid_tree);
}

__global__ void merge_horizontal_tile_boundaries_simplified(
    const std::uint8_t* image,
    int* parent,
    int width,
    int height,
    int horizontal_boundaries,
    bool complement,
    int* invalid_tree) {
    const int task =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int tasks = horizontal_boundaries * width;
    if (task >= tasks) {
        return;
    }
    const int boundary = task / width;
    const int x = task - boundary * width;
    const int bottom_y = (boundary + 1) * kTile;
    const int top = (bottom_y - 1) * width + x;
    const int bottom = top + width;
    const bool has_previous = x > 0;
    const bool has_next = x + 1 < width;
    merge_simplified_pair(
        image,
        parent,
        top,
        bottom,
        top - 1,
        bottom - 1,
        top + 1,
        bottom + 1,
        has_previous,
        has_next,
        complement,
        width * height,
        invalid_tree);
}

__global__ void repair_tile_corner_adjacencies(
    const std::uint8_t* image,
    int* parent,
    int width,
    int height,
    int vertical_boundaries,
    int horizontal_boundaries,
    bool complement,
    int* invalid_tree) {
    const int task =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int tasks = vertical_boundaries * horizontal_boundaries;
    if (task >= tasks) {
        return;
    }
    const int boundary_x = task % vertical_boundaries;
    const int boundary_y = task / vertical_boundaries;
    const int right_x = (boundary_x + 1) * kTile;
    const int bottom_y = (boundary_y + 1) * kTile;
    const int top_left = (bottom_y - 1) * width + right_x - 1;
    const int top_right = top_left + 1;
    const int bottom_left = top_left + width;
    const int bottom_right = bottom_left + 1;
    const int pixels = width * height;

    // The row-wise and column-wise simplifications are each exact for a pair
    // of 1-D trees, but independently applying both can omit an adjacency at
    // the intersection of four 2-D tiles. Reinsert the constant-size 2x2
    // corner stencil; this remains O(number of tile corners).
    connect_trees(
        image, parent, top_left, top_right,
        complement, pixels, invalid_tree);
    connect_trees(
        image, parent, bottom_left, bottom_right,
        complement, pixels, invalid_tree);
    connect_trees(
        image, parent, top_left, bottom_left,
        complement, pixels, invalid_tree);
    connect_trees(
        image, parent, top_right, bottom_right,
        complement, pixels, invalid_tree);
    connect_trees(
        image, parent, top_left, bottom_right,
        complement, pixels, invalid_tree);
    connect_trees(
        image, parent, top_right, bottom_left,
        complement, pixels, invalid_tree);
}

__global__ void flatten_tree_after_merge(
    const std::uint8_t* image,
    int* parent,
    int pixels,
    bool complement,
    int* invalid_tree) {
    const int index = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= pixels) {
        return;
    }
    int node = parent[index];
    if (node == -1) {
        return;
    }
    const int level = tree_level(image, node, complement);
    int ancestor = parent[node];
    int steps = 0;
    while (ancestor != -1 &&
           level <= tree_level(image, ancestor, complement)) {
        node = ancestor;
        ancestor = parent[node];
        if (++steps > pixels) {
            atomicExch(invalid_tree, 1);
            return;
        }
    }
    parent[index] = node;
}

__global__ void materialize_tree_leaf_mask_from_canonical_parent(
    const std::uint8_t* image,
    const int* parent,
    const std::uint8_t* has_higher_child,
    std::uint8_t* output,
    int pixels,
    bool complement) {
    const int index = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= pixels) {
        return;
    }
    const int ancestor = parent[index];
    const int representative =
        ancestor != -1 &&
        tree_level(image, ancestor, complement) ==
            tree_level(image, index, complement)
            ? ancestor
            : index;
    output[index] =
        has_higher_child[representative] == 0 ? 255 : 0;
}

static void build_one_tiled_tree_and_mask(
    TreeBuffers& buffers,
    int width,
    int height,
    bool complement,
    std::uint8_t* output) {
    const int pixels = width * height;
    const int threads = 256;
    const int blocks = (pixels + threads - 1) / threads;
    const dim3 tile_block(kTile, 1);
    const dim3 tile_grid(
        (width + kTile - 1) / kTile,
        (height + kTile - 1) / kTile);
    const int boundary_threads = 256;
    const int vertical_boundaries = (width - 1) / kTile;
    const int horizontal_boundaries = (height - 1) / kTile;

    build_local_trees_optim1d<<<tile_grid, tile_block>>>(
        buffers.image,
        buffers.parent,
        width,
        height,
        complement,
        buffers.invalid_tree);
    if (vertical_boundaries > 0) {
        const int tasks = vertical_boundaries * height;
        merge_vertical_tile_boundaries_simplified
            <<<(tasks + boundary_threads - 1) / boundary_threads,
               boundary_threads>>>(
                buffers.image,
                buffers.parent,
                width,
                height,
                vertical_boundaries,
                complement,
                buffers.invalid_tree);
    }
    if (horizontal_boundaries > 0) {
        const int tasks = horizontal_boundaries * width;
        merge_horizontal_tile_boundaries_simplified
            <<<(tasks + boundary_threads - 1) / boundary_threads,
               boundary_threads>>>(
                buffers.image,
                buffers.parent,
                width,
                height,
                horizontal_boundaries,
                complement,
                buffers.invalid_tree);
    }
    if (vertical_boundaries > 0 && horizontal_boundaries > 0) {
        const int tasks =
            vertical_boundaries * horizontal_boundaries;
        repair_tile_corner_adjacencies
            <<<(tasks + boundary_threads - 1) / boundary_threads,
               boundary_threads>>>(
                buffers.image,
                buffers.parent,
                width,
                height,
                vertical_boundaries,
                horizontal_boundaries,
                complement,
                buffers.invalid_tree);
    }
    flatten_tree_after_merge<<<blocks, threads>>>(
        buffers.image,
        buffers.parent,
        pixels,
        complement,
        buffers.invalid_tree);
    CUDA_CHECK(cudaMemset(
        buffers.has_higher_child, 0, pixels));
    mark_tree_nodes_with_higher_children<<<blocks, threads>>>(
        buffers.image,
        buffers.parent,
        buffers.has_higher_child,
        pixels,
        complement);
    materialize_tree_leaf_mask_from_canonical_parent<<<blocks, threads>>>(
        buffers.image,
        buffers.parent,
        buffers.has_higher_child,
        output,
        pixels,
        complement);
}

static void run_blin_tiled_task_complete(
    TreeBuffers& buffers,
    int width,
    int height) {
    CUDA_CHECK(cudaMemset(
        buffers.invalid_tree, 0, sizeof(int)));
    build_one_tiled_tree_and_mask(
        buffers, width, height, false, buffers.maxima);
    build_one_tiled_tree_and_mask(
        buffers, width, height, true, buffers.minima);
    CUDA_CHECK(cudaGetLastError());
}

static Timing benchmark_blin_tiled(
    TreeBuffers& buffers,
    int width,
    int height,
    int warmup,
    int repetitions,
    cv::Mat& maxima,
    cv::Mat& minima) {
    for (int index = 0; index < warmup; ++index) {
        run_blin_tiled_task_complete(buffers, width, height);
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
        run_blin_tiled_task_complete(buffers, width, height);
        CUDA_CHECK(cudaEventRecord(finish));
        CUDA_CHECK(cudaEventSynchronize(finish));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &milliseconds, start, finish));
        elapsed.push_back(milliseconds);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(finish));

    int invalid_tree = 0;
    CUDA_CHECK(cudaMemcpy(
        &invalid_tree,
        buffers.invalid_tree,
        sizeof(int),
        cudaMemcpyDeviceToHost));
    if (invalid_tree != 0) {
        throw std::runtime_error(
            "Tiled tree traversal exceeded its safety bound");
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "Usage: blin_optim1d_relaxed_flatten_extrema_v0_6_0.exe <image_dir>"
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
        std::ofstream csv(
            output_dir / "blin_optim1d_relaxed_flatten_extrema_v0_6_0.csv");
        csv
            << "algorithm,image,width,height,megapixels,warmup,repetitions,"
               "total_allocated_bytes,total_allocated_Bpp,"
               "auxiliary_bytes,auxiliary_Bpp,"
               "median_compute_ms,p95_compute_ms,compute_mpix_s,"
               "oracle_max_mismatches,oracle_min_mismatches\n";
        std::ofstream environment(output_dir / "environment.txt");
        environment
            << "harness=blin_optim1d_relaxed_flatten_extrema_v0.6.0\n"
            << "basis=Blin_et_al_TPDS_2022_sections_5_1_to_5_2\n"
            << "identity=independent_Optim1D_plus_postmerge_relaxed_flatten\n"
            << "device_name=" << properties.name << '\n'
            << "connectivity=8\n"
            << "tile=16x16_shared_memory\n"
            << "timing=resident_input_preallocated_dual_tree_and_masks\n"
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
            TreeBuffers buffers;
            allocate_tree_buffers(buffers, pixels);
            CUDA_CHECK(cudaMemcpy(
                buffers.image, image.data, pixels, cudaMemcpyHostToDevice));
            cv::Mat maxima;
            cv::Mat minima;
            const Timing timing = benchmark_blin_tiled(
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
                static_cast<std::uint64_t>(pixels) * 8ull +
                sizeof(int);
            const std::uint64_t auxiliary_bytes =
                total_bytes - static_cast<std::uint64_t>(pixels) * 3ull;
            const double megapixels = static_cast<double>(pixels) / 1e6;
            const double throughput =
                megapixels / (timing.median_ms / 1000.0);
            csv
                << quote_csv("Blin-Optim1D-RelaxedFlatten-DualTree-v0.6.0")
                << ',' << quote_csv(image_path.filename().string()) << ','
                << width << ',' << height << ',' << std::fixed
                << std::setprecision(6) << megapixels << ','
                << warmup << ',' << repetitions << ','
                << total_bytes << ','
                << static_cast<double>(total_bytes) / pixels << ','
                << auxiliary_bytes << ','
                << static_cast<double>(auxiliary_bytes) / pixels << ','
                << timing.median_ms << ',' << timing.p95_ms << ','
                << throughput << ',' << max_mismatches << ','
                << min_mismatches << '\n';
            csv.flush();
            std::cout
                << "[BLIN-OPTIM1D-RELAXED-FLATTEN-v0.6.0] "
                << image_path.filename().string()
                << " compute_ms=" << timing.median_ms
                << " mismatch=" << max_mismatches + min_mismatches
                << '\n';
            free_tree_buffers(buffers);
        }
        std::cout
            << "[BLIN-OPTIM1D-RELAXED-FLATTEN-v0.6.0-OK] images="
            << files.size()
            << " mismatches=" << total_mismatches
            << " output=" << output_dir.string() << '\n';
        return total_mismatches == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 3;
    }
}
