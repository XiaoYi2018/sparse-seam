// Task-complete independent reproduction of the sort-less concurrent
// max-tree base algorithm described by Blin et al., IEEE TPDS 2022.
//
// This is not the authors' unavailable Optim-1D source. It follows the
// paper's Algorithms 5-7: concurrent CONNECT over every forward grid edge,
// CAS-protected parent updates, and canonical flattening. The pipeline builds
// a max-tree and a min-tree (max-tree of the complemented image), identifies
// leaf flat-zones, and materializes both exact regional-extrema masks.

#define main npp_v01_main_disabled
#include "npp_labelmarkersuf_extrema_v0_1.cu"
#undef main

struct TreeBuffers {
    std::uint8_t* image = nullptr;
    int* parent = nullptr;
    std::uint8_t* has_higher_child = nullptr;
    std::uint8_t* maxima = nullptr;
    std::uint8_t* minima = nullptr;
    int* invalid_tree = nullptr;
};

struct TreePosition {
    int node;
    int parent;
};

__device__ __forceinline__ int tree_level(
    const std::uint8_t* image,
    int node,
    bool complement) {
    const int value = static_cast<int>(image[node]);
    return complement ? 255 - value : value;
}

__device__ __forceinline__ int atomic_read_parent(const int* address) {
    return atomicAdd(const_cast<int*>(address), 0);
}

__device__ __forceinline__ bool tree_precedes(
    const std::uint8_t* image,
    int first,
    int second,
    bool complement) {
    const int first_level = tree_level(image, first, complement);
    const int second_level = tree_level(image, second, complement);
    return first_level < second_level ||
           (first_level == second_level && first > second);
}

__device__ __forceinline__ TreePosition find_peak_root(
    const std::uint8_t* image,
    const int* parent,
    int node,
    int level,
    bool complement,
    int pixels,
    int* invalid_tree) {
    int ancestor = atomic_read_parent(parent + node);
    int steps = 0;
    while (ancestor != -1 &&
           level <= tree_level(image, ancestor, complement)) {
        node = ancestor;
        ancestor = atomic_read_parent(parent + node);
        if (++steps > pixels) {
            atomicExch(invalid_tree, 1);
            return TreePosition{node, -1};
        }
    }
    return TreePosition{node, ancestor};
}

__device__ __forceinline__ TreePosition find_level_root(
    const std::uint8_t* image,
    const int* parent,
    int node,
    bool complement,
    int pixels,
    int* invalid_tree) {
    return find_peak_root(
        image,
        parent,
        node,
        tree_level(image, node, complement),
        complement,
        pixels,
        invalid_tree);
}

__device__ __forceinline__ void connect_trees(
    const std::uint8_t* image,
    int* parent,
    int first,
    int second,
    bool complement,
    int pixels,
    int* invalid_tree) {
    int retries = 0;
    while (second != -1) {
        if (tree_level(image, second, complement) <
            tree_level(image, first, complement)) {
            const int temporary = first;
            first = second;
            second = temporary;
        }

        TreePosition first_position = find_level_root(
            image, parent, first, complement, pixels, invalid_tree);
        TreePosition second_position = find_peak_root(
            image,
            parent,
            second,
            tree_level(image, first_position.node, complement),
            complement,
            pixels,
            invalid_tree);
        first = first_position.node;
        second = second_position.node;

        if (tree_precedes(
                image, second, first, complement)) {
            int temporary = first;
            first = second;
            second = temporary;
            temporary = first_position.parent;
            first_position.parent = second_position.parent;
            second_position.parent = temporary;
        }
        if (first == second) {
            return;
        }

        const int observed = atomicCAS(
            parent + second, second_position.parent, first);
        if (observed == second_position.parent) {
            second = observed;
        }
        if (++retries > pixels * 2) {
            atomicExch(invalid_tree, 1);
            return;
        }
    }
}

__global__ void initialize_tree(int* parent, int pixels) {
    const int index = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (index < pixels) {
        parent[index] = -1;
    }
}

__global__ void connect_forward_8_neighbour_edges(
    const std::uint8_t* image,
    int* parent,
    int width,
    int height,
    bool complement,
    int* invalid_tree) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const int pixels = width * height;
    const int index = y * width + x;
    if (x + 1 < width) {
        connect_trees(
            image, parent, index, index + 1,
            complement, pixels, invalid_tree);
    }
    if (y + 1 < height) {
        if (x > 0) {
            connect_trees(
                image, parent, index, index + width - 1,
                complement, pixels, invalid_tree);
        }
        connect_trees(
            image, parent, index, index + width,
            complement, pixels, invalid_tree);
        if (x + 1 < width) {
            connect_trees(
                image, parent, index, index + width + 1,
                complement, pixels, invalid_tree);
        }
    }
}

__global__ void flatten_tree(
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
    const int current_parent = atomic_read_parent(parent + index);
    if (current_parent == -1) {
        return;
    }
    const TreePosition canonical = find_level_root(
        image,
        parent,
        current_parent,
        complement,
        pixels,
        invalid_tree);
    parent[index] = canonical.node;
}

__global__ void mark_tree_nodes_with_higher_children(
    const std::uint8_t* image,
    const int* parent,
    std::uint8_t* has_higher_child,
    int pixels,
    bool complement) {
    const int index = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= pixels) {
        return;
    }
    const int ancestor = parent[index];
    if (ancestor != -1 &&
        tree_level(image, ancestor, complement) <
            tree_level(image, index, complement)) {
        has_higher_child[ancestor] = 1;
    }
}

__device__ __forceinline__ int flat_zone_representative(
    const std::uint8_t* image,
    const int* parent,
    int node,
    bool complement,
    int pixels,
    int* invalid_tree) {
    return find_level_root(
        image,
        parent,
        node,
        complement,
        pixels,
        invalid_tree).node;
}

__global__ void materialize_tree_leaf_mask(
    const std::uint8_t* image,
    const int* parent,
    const std::uint8_t* has_higher_child,
    std::uint8_t* output,
    int pixels,
    bool complement,
    int* invalid_tree) {
    const int index = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= pixels) {
        return;
    }
    const int representative = flat_zone_representative(
        image,
        parent,
        index,
        complement,
        pixels,
        invalid_tree);
    output[index] =
        has_higher_child[representative] == 0 ? 255 : 0;
}

static void allocate_tree_buffers(TreeBuffers& buffers, int pixels) {
    CUDA_CHECK(cudaMalloc(&buffers.image, pixels));
    CUDA_CHECK(cudaMalloc(
        &buffers.parent,
        static_cast<std::size_t>(pixels) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buffers.has_higher_child, pixels));
    CUDA_CHECK(cudaMalloc(&buffers.maxima, pixels));
    CUDA_CHECK(cudaMalloc(&buffers.minima, pixels));
    CUDA_CHECK(cudaMalloc(&buffers.invalid_tree, sizeof(int)));
}

static void free_tree_buffers(TreeBuffers& buffers) {
    cudaFree(buffers.image);
    cudaFree(buffers.parent);
    cudaFree(buffers.has_higher_child);
    cudaFree(buffers.maxima);
    cudaFree(buffers.minima);
    cudaFree(buffers.invalid_tree);
    buffers = TreeBuffers{};
}

static void build_one_tree_and_mask(
    TreeBuffers& buffers,
    int width,
    int height,
    bool complement,
    std::uint8_t* output) {
    const int pixels = width * height;
    const int threads = 256;
    const int blocks = (pixels + threads - 1) / threads;
    const dim3 block_2d(16, 16);
    const dim3 grid_2d(
        (width + block_2d.x - 1) / block_2d.x,
        (height + block_2d.y - 1) / block_2d.y);
    initialize_tree<<<blocks, threads>>>(buffers.parent, pixels);
    CUDA_CHECK(cudaMemset(
        buffers.has_higher_child, 0, pixels));
    connect_forward_8_neighbour_edges<<<grid_2d, block_2d>>>(
        buffers.image,
        buffers.parent,
        width,
        height,
        complement,
        buffers.invalid_tree);
    flatten_tree<<<blocks, threads>>>(
        buffers.image,
        buffers.parent,
        pixels,
        complement,
        buffers.invalid_tree);
    mark_tree_nodes_with_higher_children<<<blocks, threads>>>(
        buffers.image,
        buffers.parent,
        buffers.has_higher_child,
        pixels,
        complement);
    materialize_tree_leaf_mask<<<blocks, threads>>>(
        buffers.image,
        buffers.parent,
        buffers.has_higher_child,
        output,
        pixels,
        complement,
        buffers.invalid_tree);
}

static void run_blin_style_task_complete(
    TreeBuffers& buffers,
    int width,
    int height) {
    CUDA_CHECK(cudaMemset(
        buffers.invalid_tree, 0, sizeof(int)));
    build_one_tree_and_mask(
        buffers, width, height, false, buffers.maxima);
    build_one_tree_and_mask(
        buffers, width, height, true, buffers.minima);
    CUDA_CHECK(cudaGetLastError());
}

static Timing benchmark_blin_style(
    TreeBuffers& buffers,
    int width,
    int height,
    int warmup,
    int repetitions,
    cv::Mat& maxima,
    cv::Mat& minima) {
    for (int index = 0; index < warmup; ++index) {
        run_blin_style_task_complete(buffers, width, height);
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
        run_blin_style_task_complete(buffers, width, height);
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
            "Tree traversal exceeded its safety bound");
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

#ifndef BLIN_V01_COMMON_ONLY
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "Usage: blin_sortless_maxtree_extrema_v0_1.exe <image_dir>"
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
            output_dir / "blin_sortless_maxtree_extrema_v0_1.csv");
        csv
            << "algorithm,image,width,height,megapixels,warmup,repetitions,"
               "total_allocated_bytes,total_allocated_Bpp,"
               "auxiliary_bytes,auxiliary_Bpp,"
               "median_compute_ms,p95_compute_ms,compute_mpix_s,"
               "oracle_max_mismatches,oracle_min_mismatches\n";
        std::ofstream environment(output_dir / "environment.txt");
        environment
            << "harness=blin_sortless_maxtree_extrema_v0.1\n"
            << "basis=Blin_et_al_TPDS_2022_algorithms_5_to_7\n"
            << "identity=independent_Base_reproduction_not_author_Optim1D\n"
            << "device_name=" << properties.name << '\n'
            << "connectivity=8\n"
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
            const Timing timing = benchmark_blin_style(
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
                << quote_csv(
                       "Blin-style-Sortless-Base-DualTree-v0.1")
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
                << "[BLIN-BASE-v0.1] "
                << image_path.filename().string()
                << " compute_ms=" << timing.median_ms
                << " mismatch=" << max_mismatches + min_mismatches
                << '\n';
            free_tree_buffers(buffers);
        }
        std::cout
            << "[BLIN-BASE-v0.1-OK] images=" << files.size()
            << " mismatches=" << total_mismatches
            << " output=" << output_dir.string() << '\n';
        return total_mismatches == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 3;
    }
}
#endif
