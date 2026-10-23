// Exact task-complete production-library baseline for 8-neighbour regional
// maxima and minima.
//
// NVIDIA NPP 12.6 LabelMarkersUF is used as the initial flat-zone labeler.
// The project oracle exposed a reproducible nppiNormInf split of equal-valued
// diagonal neighbours. Therefore this version performs a transparent exact
// repair: it scans the four forward 8-neighbour edges and unions NPP labels
// only when the source pixels are equal. Repair, boundary reduction, both
// output masks, NPP scratch, and every persistent buffer are timed/accounted.

#define main npp_v01_main_disabled
#include "npp_labelmarkersuf_extrema_v0_1.cu"
#undef main

struct RepairBuffers {
    Npp8u* image = nullptr;
    Npp32u* labels = nullptr;
    std::uint32_t* repair_parent = nullptr;
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

__device__ __forceinline__ std::uint32_t repair_find(
    std::uint32_t* parent,
    std::uint32_t node) {
    std::uint32_t current = node;
    while (parent[current] != current) {
        current = parent[current];
    }
    return current;
}

__device__ __forceinline__ void repair_union(
    std::uint32_t* parent,
    std::uint32_t first,
    std::uint32_t second) {
    while (true) {
        first = repair_find(parent, first);
        second = repair_find(parent, second);
        if (first == second) {
            return;
        }
        const std::uint32_t lower = min(first, second);
        const std::uint32_t higher = max(first, second);
        const std::uint32_t previous =
            atomicCAS(parent + higher, higher, lower);
        if (previous == higher) {
            return;
        }
    }
}

__global__ void initialize_repair_parent(
    std::uint32_t* parent,
    int entries) {
    const int index = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (index < entries) {
        parent[index] = static_cast<std::uint32_t>(index);
    }
}

__global__ void repair_equal_label_edges(
    const Npp8u* image,
    const Npp32u* labels,
    std::uint32_t* parent,
    int* invalid_label,
    int width,
    int height) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const int pixels = width * height;
    const int index = y * width + x;
    const Npp32u source_label = labels[index];
    if (source_label > static_cast<Npp32u>(pixels)) {
        atomicExch(invalid_label, 1);
        return;
    }
    const Npp8u value = image[index];
    const int dx[4] = {1, -1, 0, 1};
    const int dy[4] = {0, 1, 1, 1};
    #pragma unroll
    for (int edge = 0; edge < 4; ++edge) {
        const int nx = x + dx[edge];
        const int ny = y + dy[edge];
        if (nx < 0 || nx >= width || ny >= height) {
            continue;
        }
        const int neighbour = ny * width + nx;
        if (image[neighbour] != value) {
            continue;
        }
        const Npp32u neighbour_label = labels[neighbour];
        if (neighbour_label > static_cast<Npp32u>(pixels)) {
            atomicExch(invalid_label, 1);
            continue;
        }
        if (source_label != neighbour_label) {
            repair_union(parent, source_label, neighbour_label);
        }
    }
}

__global__ void compress_repair_parent(
    std::uint32_t* parent,
    int entries) {
    const int index = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= entries) {
        return;
    }
    const std::uint32_t root =
        repair_find(parent, static_cast<std::uint32_t>(index));
    parent[index] = root;
}

__global__ void reduce_flags_by_repaired_label(
    const Npp32u* labels,
    const std::uint32_t* repair_parent,
    const std::uint8_t* has_greater,
    const std::uint8_t* has_smaller,
    std::uint32_t* root_any_greater,
    std::uint32_t* root_any_smaller,
    int pixels) {
    const int index = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= pixels) {
        return;
    }
    const std::uint32_t root = repair_parent[labels[index]];
    if (has_greater[index] != 0) {
        atomicOr(root_any_greater + root, 1u);
    }
    if (has_smaller[index] != 0) {
        atomicOr(root_any_smaller + root, 1u);
    }
}

__global__ void materialize_repaired_masks(
    const Npp32u* labels,
    const std::uint32_t* repair_parent,
    const std::uint32_t* root_any_greater,
    const std::uint32_t* root_any_smaller,
    std::uint8_t* maxima,
    std::uint8_t* minima,
    int pixels) {
    const int index = static_cast<int>(
        blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= pixels) {
        return;
    }
    const std::uint32_t root = repair_parent[labels[index]];
    maxima[index] = root_any_greater[root] == 0 ? 255 : 0;
    minima[index] = root_any_smaller[root] == 0 ? 255 : 0;
}

static void allocate_repair_buffers(
    RepairBuffers& buffers,
    int width,
    int height) {
    const int pixels = width * height;
    const std::size_t labels_bytes =
        static_cast<std::size_t>(pixels) * sizeof(Npp32u);
    const std::size_t roots_bytes =
        static_cast<std::size_t>(pixels + 1) * sizeof(std::uint32_t);
    const NppiSize roi{width, height};
    NPP_CHECK(nppiLabelMarkersUFGetBufferSize_32u_C1R(
        roi, &buffers.scratch_bytes));
    if (buffers.scratch_bytes < 1) {
        throw std::runtime_error("NPP returned an invalid scratch size");
    }
    CUDA_CHECK(cudaMalloc(&buffers.image, pixels));
    CUDA_CHECK(cudaMalloc(&buffers.labels, labels_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.repair_parent, roots_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.has_greater, pixels));
    CUDA_CHECK(cudaMalloc(&buffers.has_smaller, pixels));
    CUDA_CHECK(cudaMalloc(&buffers.root_any_greater, roots_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.root_any_smaller, roots_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.maxima, pixels));
    CUDA_CHECK(cudaMalloc(&buffers.minima, pixels));
    CUDA_CHECK(cudaMalloc(&buffers.npp_scratch, buffers.scratch_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.invalid_label, sizeof(int)));
}

static void free_repair_buffers(RepairBuffers& buffers) {
    cudaFree(buffers.image);
    cudaFree(buffers.labels);
    cudaFree(buffers.repair_parent);
    cudaFree(buffers.has_greater);
    cudaFree(buffers.has_smaller);
    cudaFree(buffers.root_any_greater);
    cudaFree(buffers.root_any_smaller);
    cudaFree(buffers.maxima);
    cudaFree(buffers.minima);
    cudaFree(buffers.npp_scratch);
    cudaFree(buffers.invalid_label);
    buffers = RepairBuffers{};
}

static void run_repaired_task_complete(
    RepairBuffers& buffers,
    int width,
    int height) {
    const int pixels = width * height;
    const int entries = pixels + 1;
    const int threads = 256;
    const int pixel_blocks = (pixels + threads - 1) / threads;
    const int entry_blocks = (entries + threads - 1) / threads;
    const dim3 block_2d(16, 16);
    const dim3 grid_2d(
        (width + block_2d.x - 1) / block_2d.x,
        (height + block_2d.y - 1) / block_2d.y);
    const NppiSize roi{width, height};
    const std::size_t root_bytes =
        static_cast<std::size_t>(entries) * sizeof(std::uint32_t);

    CUDA_CHECK(cudaMemset(buffers.root_any_greater, 0, root_bytes));
    CUDA_CHECK(cudaMemset(buffers.root_any_smaller, 0, root_bytes));
    CUDA_CHECK(cudaMemset(buffers.invalid_label, 0, sizeof(int)));
    initialize_repair_parent<<<entry_blocks, threads>>>(
        buffers.repair_parent, entries);
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
    repair_equal_label_edges<<<grid_2d, block_2d>>>(
        buffers.image,
        buffers.labels,
        buffers.repair_parent,
        buffers.invalid_label,
        width,
        height);
    compress_repair_parent<<<entry_blocks, threads>>>(
        buffers.repair_parent, entries);
    reduce_flags_by_repaired_label<<<pixel_blocks, threads>>>(
        buffers.labels,
        buffers.repair_parent,
        buffers.has_greater,
        buffers.has_smaller,
        buffers.root_any_greater,
        buffers.root_any_smaller,
        pixels);
    materialize_repaired_masks<<<pixel_blocks, threads>>>(
        buffers.labels,
        buffers.repair_parent,
        buffers.root_any_greater,
        buffers.root_any_smaller,
        buffers.maxima,
        buffers.minima,
        pixels);
    CUDA_CHECK(cudaGetLastError());
}

static Timing benchmark_repaired(
    RepairBuffers& buffers,
    int width,
    int height,
    int warmup,
    int repetitions,
    cv::Mat& maxima,
    cv::Mat& minima) {
    for (int index = 0; index < warmup; ++index) {
        run_repaired_task_complete(buffers, width, height);
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
        run_repaired_task_complete(buffers, width, height);
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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "Usage: npp_labelmarkersuf_exact_repair_v0_3.exe <image_dir>"
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
            output_dir / "npp_labelmarkersuf_exact_repair_v0_3.csv");
        csv
            << "algorithm,image,width,height,megapixels,warmup,repetitions,"
               "npp_scratch_bytes,total_allocated_bytes,total_allocated_Bpp,"
               "auxiliary_bytes,auxiliary_Bpp,"
               "median_compute_ms,p95_compute_ms,compute_mpix_s,"
               "oracle_max_mismatches,oracle_min_mismatches\n";
        std::ofstream environment(output_dir / "environment.txt");
        environment
            << "harness=npp_labelmarkersuf_exact_repair_v0.3\n"
            << "initial_labeling=NVIDIA_NPP_12.6_"
               "LabelMarkersUF_8u32u_C1R\n"
            << "documented_basis=Chen_et_al_optimized_GPU_union_find\n"
            << "repair=union_distinct_NPP_labels_on_equal_forward_"
               "8_neighbour_edges\n"
            << "device_name=" << properties.name << '\n'
            << "connectivity=8\n"
            << "timing=resident_input_preallocated_task_complete\n"
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
            RepairBuffers buffers;
            allocate_repair_buffers(buffers, width, height);
            CUDA_CHECK(cudaMemcpy(
                buffers.image, image.data, pixels, cudaMemcpyHostToDevice));
            cv::Mat maxima;
            cv::Mat minima;
            const Timing timing = benchmark_repaired(
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
                static_cast<std::uint64_t>(pixels) * 21ull +
                static_cast<std::uint64_t>(buffers.scratch_bytes) +
                3ull * sizeof(std::uint32_t) + sizeof(int);
            const std::uint64_t auxiliary_bytes =
                total_bytes - static_cast<std::uint64_t>(pixels) * 3ull;
            const double megapixels = static_cast<double>(pixels) / 1e6;
            const double throughput =
                megapixels / (timing.median_ms / 1000.0);
            csv
                << quote_csv(
                       "NPP-LabelMarkersUF-ExactRepair-Extrema-v0.3")
                << ',' << quote_csv(image_path.filename().string()) << ','
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
                << "[NPP-UF-REPAIR-v0.3] "
                << image_path.filename().string()
                << " compute_ms=" << timing.median_ms
                << " scratch=" << buffers.scratch_bytes
                << " mismatch=" << max_mismatches + min_mismatches
                << '\n';
            free_repair_buffers(buffers);
        }
        std::cout
            << "[NPP-UF-REPAIR-v0.3-OK] images=" << files.size()
            << " mismatches=" << total_mismatches
            << " output=" << output_dir.string() << '\n';
        return total_mismatches == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 3;
    }
}
