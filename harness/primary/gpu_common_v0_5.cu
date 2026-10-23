// Unified task-complete CUDA benchmark v0.5.
//
// This translation unit compiles immutable source snapshots of:
//   1) Global-UF and FlatZone-OLE from gpu_common_v0.1;
//   2) Sparse-Seam v0.9 variant 1 kernels at 32x32 tiles.
//
// It provides one process, one dataset traversal, identical warm-up/repetition
// settings, resident-compute and reusable-allocation host-to-host timing, exact
// output validation, and separate total/auxiliary memory accounting.

#define main common_v01_disabled_main
#define fs common_fs
#define DeviceBuffers CommonDeviceBuffers
#define TimingSummary CommonTimingSummary
#define ends_with_ci common_ends_with_ci
#define list_images common_list_images
#define allocate_buffers common_allocate_buffers
#define free_buffers common_free_buffers
#define compare_masks common_compare_masks
#define quote_csv common_quote_csv
#define write_row common_write_row
#include "snapshot_gpu_benchmark_globaluf_ole_v0_1.cu"
#undef write_row
#undef quote_csv
#undef compare_masks
#undef free_buffers
#undef allocate_buffers
#undef list_images
#undef ends_with_ci
#undef TimingSummary
#undef DeviceBuffers
#undef fs
#undef main
#undef CUDA_CHECK

#define main sparse_v09_disabled_main
#define fs sparse_fs
#define DeviceBuffers SparseDeviceBuffers
#define ends_with_ci sparse_ends_with_ci
#define list_images sparse_list_images
#define allocate_buffers sparse_allocate_buffers
#define free_buffers sparse_free_buffers
#define run_sparse_seam sparse_original_run_sparse_seam
#define compare_masks sparse_compare_masks
#include "snapshot_sparse_seam_cuda_v0_9_variant1.cu"
#undef compare_masks
#undef run_sparse_seam
#undef free_buffers
#undef allocate_buffers
#undef list_images
#undef ends_with_ci
#undef DeviceBuffers
#undef fs
#undef main
#undef CUDA_CHECK

#include <numeric>

#define CUDA_CHECK_V2(call)                                                     \
    do {                                                                        \
        const cudaError_t code = (call);                                        \
        if (code != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error: ") +             \
                                     cudaGetErrorString(code));                  \
        }                                                                       \
    } while (0)

namespace unified_v05 {

namespace fs = std::filesystem;

struct Timing {
    double median_compute_ms = 0.0;
    double p95_compute_ms = 0.0;
    double median_h2h_ms = 0.0;
    double p95_h2h_ms = 0.0;
    double median_iterations = 1.0;
    double p95_iterations = 1.0;
};

struct Memory {
    std::uint64_t total_bytes = 0;
    std::uint64_t auxiliary_bytes = 0;
};

static double percentile_v02(
    std::vector<double> values,
    double probability
) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double position =
        probability * static_cast<double>(values.size() - 1);
    const std::size_t lower =
        static_cast<std::size_t>(std::floor(position));
    const std::size_t upper =
        static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) +
           values[upper] * fraction;
}

static std::uint64_t compare_masks_v02(
    const cv::Mat& first,
    const cv::Mat& second
) {
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

static std::string quote_csv_v02(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (const char character : value) {
        if (character == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(character);
    }
    escaped.push_back('"');
    return escaped;
}

static int run_sparse_resident(
    SparseDeviceBuffers& buffers,
    int width,
    int height
) {
    const int pixels = width * height;
    const int tiles_x = (width + TILE - 1) / TILE;
    const int tiles_y = (height + TILE - 1) / TILE;
    const int fragment_capacity =
        tiles_x * tiles_y * SLOTS_PER_TILE;
    CUDA_CHECK_V2(cudaMemset(
        buffers.error_flag, 0, sizeof(int)));

    const dim3 tile_grid(tiles_x, tiles_y);
    constexpr int threads = 256;
    const int pixel_blocks = (pixels + threads - 1) / threads;
    const int seam_entries =
        std::max(0, tiles_x - 1) * height +
        std::max(0, tiles_y - 1) * width;
    const int seam_blocks =
        (seam_entries + threads - 1) / threads;
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
    CUDA_CHECK_V2(cudaGetLastError());
    return 1;
}

using CommonAlgorithm =
    int (*)(CommonDeviceBuffers&, int, int, int);

static Timing benchmark_common_v02(
    CommonAlgorithm algorithm,
    CommonDeviceBuffers& buffers,
    const cv::Mat& image,
    int block,
    int warmup,
    int repetitions,
    cv::Mat& last_maxima,
    cv::Mat& last_minima
) {
    const int width = image.cols;
    const int height = image.rows;
    const int pixels = width * height;
    const std::size_t pixel_bytes =
        static_cast<std::size_t>(pixels) * sizeof(std::uint8_t);
    CUDA_CHECK_V2(cudaMemcpy(
        buffers.image,
        image.data,
        pixel_bytes,
        cudaMemcpyHostToDevice));

    for (int index = 0; index < warmup; ++index) {
        algorithm(buffers, width, height, block);
        CUDA_CHECK_V2(cudaDeviceSynchronize());
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK_V2(cudaEventCreate(&start));
    CUDA_CHECK_V2(cudaEventCreate(&stop));
    std::vector<double> compute_times;
    std::vector<double> h2h_times;
    std::vector<double> iterations;
    compute_times.reserve(static_cast<std::size_t>(repetitions));
    h2h_times.reserve(static_cast<std::size_t>(repetitions));
    iterations.reserve(static_cast<std::size_t>(repetitions));

    for (int repetition = 0; repetition < repetitions; ++repetition) {
        CUDA_CHECK_V2(cudaEventRecord(start));
        const int iteration_count =
            algorithm(buffers, width, height, block);
        CUDA_CHECK_V2(cudaEventRecord(stop));
        CUDA_CHECK_V2(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK_V2(cudaEventElapsedTime(
            &elapsed_ms, start, stop));
        compute_times.push_back(static_cast<double>(elapsed_ms));
        iterations.push_back(static_cast<double>(iteration_count));
    }

    last_maxima.create(height, width, CV_8U);
    last_minima.create(height, width, CV_8U);
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        const auto begin = std::chrono::high_resolution_clock::now();
        CUDA_CHECK_V2(cudaMemcpy(
            buffers.image,
            image.data,
            pixel_bytes,
            cudaMemcpyHostToDevice));
        algorithm(buffers, width, height, block);
        CUDA_CHECK_V2(cudaMemcpy(
            last_maxima.data,
            buffers.maxima,
            pixel_bytes,
            cudaMemcpyDeviceToHost));
        CUDA_CHECK_V2(cudaMemcpy(
            last_minima.data,
            buffers.minima,
            pixel_bytes,
            cudaMemcpyDeviceToHost));
        const auto end = std::chrono::high_resolution_clock::now();
        h2h_times.push_back(
            std::chrono::duration<double, std::milli>(
                end - begin).count());
    }
    CUDA_CHECK_V2(cudaEventDestroy(start));
    CUDA_CHECK_V2(cudaEventDestroy(stop));

    Timing result;
    result.median_compute_ms =
        percentile_v02(compute_times, 0.5);
    result.p95_compute_ms =
        percentile_v02(compute_times, 0.95);
    result.median_h2h_ms =
        percentile_v02(h2h_times, 0.5);
    result.p95_h2h_ms =
        percentile_v02(h2h_times, 0.95);
    result.median_iterations =
        percentile_v02(iterations, 0.5);
    result.p95_iterations =
        percentile_v02(iterations, 0.95);
    return result;
}

static Timing benchmark_sparse_v02(
    SparseDeviceBuffers& buffers,
    const cv::Mat& image,
    int warmup,
    int repetitions,
    cv::Mat& last_maxima,
    cv::Mat& last_minima
) {
    const int width = image.cols;
    const int height = image.rows;
    const int pixels = width * height;
    const std::size_t pixel_bytes =
        static_cast<std::size_t>(pixels) * sizeof(std::uint8_t);
    CUDA_CHECK_V2(cudaMemcpy(
        buffers.image,
        image.data,
        pixel_bytes,
        cudaMemcpyHostToDevice));

    for (int index = 0; index < warmup; ++index) {
        run_sparse_resident(buffers, width, height);
        CUDA_CHECK_V2(cudaDeviceSynchronize());
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK_V2(cudaEventCreate(&start));
    CUDA_CHECK_V2(cudaEventCreate(&stop));
    std::vector<double> compute_times;
    std::vector<double> h2h_times;
    compute_times.reserve(static_cast<std::size_t>(repetitions));
    h2h_times.reserve(static_cast<std::size_t>(repetitions));

    for (int repetition = 0; repetition < repetitions; ++repetition) {
        CUDA_CHECK_V2(cudaEventRecord(start));
        run_sparse_resident(buffers, width, height);
        CUDA_CHECK_V2(cudaEventRecord(stop));
        CUDA_CHECK_V2(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0f;
        CUDA_CHECK_V2(cudaEventElapsedTime(
            &elapsed_ms, start, stop));
        compute_times.push_back(static_cast<double>(elapsed_ms));
    }

    last_maxima.create(height, width, CV_8U);
    last_minima.create(height, width, CV_8U);
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        const auto begin = std::chrono::high_resolution_clock::now();
        CUDA_CHECK_V2(cudaMemcpy(
            buffers.image,
            image.data,
            pixel_bytes,
            cudaMemcpyHostToDevice));
        run_sparse_resident(buffers, width, height);
        CUDA_CHECK_V2(cudaMemcpy(
            last_maxima.data,
            buffers.maxima,
            pixel_bytes,
            cudaMemcpyDeviceToHost));
        CUDA_CHECK_V2(cudaMemcpy(
            last_minima.data,
            buffers.minima,
            pixel_bytes,
            cudaMemcpyDeviceToHost));
        const auto end = std::chrono::high_resolution_clock::now();
        h2h_times.push_back(
            std::chrono::duration<double, std::milli>(
                end - begin).count());
    }
    CUDA_CHECK_V2(cudaEventDestroy(start));
    CUDA_CHECK_V2(cudaEventDestroy(stop));

    int invariant = 0;
    CUDA_CHECK_V2(cudaMemcpy(
        &invariant,
        buffers.error_flag,
        sizeof(int),
        cudaMemcpyDeviceToHost));
    if (invariant != 0) {
        throw std::runtime_error(
            "Sparse-seam invariant flag=" +
            std::to_string(invariant));
    }

    Timing result;
    result.median_compute_ms =
        percentile_v02(compute_times, 0.5);
    result.p95_compute_ms =
        percentile_v02(compute_times, 0.95);
    result.median_h2h_ms =
        percentile_v02(h2h_times, 0.5);
    result.p95_h2h_ms =
        percentile_v02(h2h_times, 0.95);
    return result;
}

static Memory common_memory_v02(int pixels) {
    const std::uint64_t count =
        static_cast<std::uint64_t>(pixels);
    Memory result;
    result.total_bytes = count * 17ull + sizeof(int);
    result.auxiliary_bytes = count * 14ull + sizeof(int);
    return result;
}

static Memory sparse_memory_v02(
    int pixels,
    int fragment_capacity,
    int tile_count
) {
    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(pixels);
    const std::uint64_t fragment_count =
        static_cast<std::uint64_t>(fragment_capacity);
    const std::uint64_t tiles =
        static_cast<std::uint64_t>(tile_count);
    Memory result;
    result.total_bytes =
        pixel_count * 4ull + fragment_count * 16ull +
        tiles * sizeof(std::uint16_t) + sizeof(int);
    result.auxiliary_bytes =
        pixel_count + fragment_count * 16ull +
        tiles * sizeof(std::uint16_t) + sizeof(int);
    return result;
}

static void write_row_v02(
    std::ofstream& csv,
    const std::string& algorithm,
    const fs::path& image_path,
    int width,
    int height,
    int warmup,
    int repetitions,
    int block,
    const Memory& memory,
    const Timing& timing,
    std::uint64_t oracle_max_mismatches,
    std::uint64_t oracle_min_mismatches,
    std::uint64_t cross_max_mismatches,
    std::uint64_t cross_min_mismatches
) {
    const double pixels =
        static_cast<double>(width) * static_cast<double>(height);
    const double megapixels = pixels / 1e6;
    const double compute_mpix_s =
        megapixels / (timing.median_compute_ms / 1000.0);
    const double h2h_mpix_s =
        megapixels / (timing.median_h2h_ms / 1000.0);
    csv << quote_csv_v02(algorithm) << ','
        << quote_csv_v02(image_path.filename().string()) << ','
        << width << ',' << height << ',' << std::fixed
        << std::setprecision(6) << megapixels << ','
        << warmup << ',' << repetitions << ',' << block << ','
        << memory.total_bytes << ','
        << static_cast<double>(memory.total_bytes) / pixels << ','
        << memory.auxiliary_bytes << ','
        << static_cast<double>(memory.auxiliary_bytes) / pixels << ','
        << timing.median_compute_ms << ','
        << timing.p95_compute_ms << ','
        << compute_mpix_s << ','
        << timing.median_h2h_ms << ','
        << timing.p95_h2h_ms << ','
        << h2h_mpix_s << ','
        << timing.median_iterations << ','
        << timing.p95_iterations << ','
        << oracle_max_mismatches << ','
        << oracle_min_mismatches << ','
        << cross_max_mismatches << ','
        << cross_min_mismatches << '\n';
}

}  // namespace unified_v05

int main(int argc, char** argv) {
    using namespace unified_v05;
    if (argc < 2) {
        std::cerr
            << "Usage: gpu_common_v0_5.exe <image_dir> --output=DIR"
               " [--expected-max=DIR --expected-min=DIR]"
               " [--device=0] [--block=16] [--warmup=5] [--reps=30]"
               " [--order=gos|sog]\n";
        return 2;
    }

    try {
        const fs::path image_dir = argv[1];
        fs::path output_dir;
        fs::path expected_max_dir;
        fs::path expected_min_dir;
        int device = 0;
        int block = 16;
        int warmup = 5;
        int repetitions = 30;
        std::string execution_order = "gos";
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
            } else if (argument.rfind("--block=", 0) == 0) {
                block = std::stoi(argument.substr(8));
            } else if (argument.rfind("--warmup=", 0) == 0) {
                warmup = std::stoi(argument.substr(9));
            } else if (argument.rfind("--reps=", 0) == 0) {
                repetitions = std::stoi(argument.substr(7));
            } else if (argument.rfind("--order=", 0) == 0) {
                execution_order = argument.substr(8);
            }
        }
        if (output_dir.empty()) {
            throw std::runtime_error("Missing --output directory");
        }
        if (expected_max_dir.empty() != expected_min_dir.empty()) {
            throw std::runtime_error(
                "Expected max/min directories must be supplied together");
        }
        if (block < 4 || block > 32 ||
            warmup < 0 || repetitions < 1) {
            throw std::runtime_error("Invalid benchmark option");
        }
        if (execution_order != "gos" &&
            execution_order != "sog") {
            throw std::runtime_error(
                "Execution order must be gos or sog");
        }

        CUDA_CHECK_V2(cudaSetDevice(device));
        cudaDeviceProp properties{};
        CUDA_CHECK_V2(cudaGetDeviceProperties(&properties, device));
        fs::create_directories(output_dir);
        std::ofstream csv(output_dir / "gpu_common_v0_5.csv");
        if (!csv) {
            throw std::runtime_error("Cannot create benchmark CSV");
        }
        csv
            << "algorithm,image,width,height,megapixels,warmup,repetitions,"
               "block,total_allocated_bytes,total_allocated_Bpp,"
               "auxiliary_bytes,auxiliary_Bpp,"
               "median_compute_ms,p95_compute_ms,compute_mpix_s,"
               "median_h2h_ms,p95_h2h_ms,h2h_mpix_s,"
               "median_iterations,p95_iterations,"
               "oracle_max_mismatches,oracle_min_mismatches,"
               "cross_max_mismatches,cross_min_mismatches\n";

        std::ofstream environment(output_dir / "environment.txt");
        environment
            << "harness=gpu_common_v0.5\n"
            << "device_name=" << properties.name << '\n'
            << "compute_capability=" << properties.major << '.'
            << properties.minor << '\n'
            << "global_memory_bytes=" << properties.totalGlobalMem << '\n'
            << "cuda_runtime=" << CUDART_VERSION << '\n'
            << "block=" << block << '\n'
            << "warmup=" << warmup << '\n'
            << "repetitions=" << repetitions << '\n'
            << "execution_order=" << execution_order << '\n'
            << "timing_compute=preallocated,resident_input,task_complete\n"
            << "timing_h2h=reused_allocation,H2D+compute+D2H_two_masks\n"
            << "auxiliary_definition=all_device_buffers_except_input_and_"
               "two_output_masks\n"
            << "source_common_v0.1_sha256="
               "FE817F2C06F557D2313FCF36E3CC875409B9C73D513129455F67021098A1C8CD\n"
            << "source_sparse_v0.9_sha256="
               "B8F4866E0403C68798C89D668ADBD4972B847FE211752C2348B894951B0CC687\n"
            << "sparse_tile=32\n"
            << "sparse_classification_variant=1\n"
            << "sparse_initialization=fused_into_tile_classification\n";

        const std::vector<fs::path> files =
            common_list_images(image_dir);
        std::uint64_t total_oracle_mismatches = 0;
        std::uint64_t total_cross_mismatches = 0;

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
            const int tiles_x = (width + TILE - 1) / TILE;
            const int tiles_y = (height + TILE - 1) / TILE;
            const int tile_count = tiles_x * tiles_y;
            const int fragment_capacity =
                tile_count * SLOTS_PER_TILE;

            CommonDeviceBuffers common_buffers;
            SparseDeviceBuffers sparse_buffers;
            common_allocate_buffers(common_buffers, pixels);
            sparse_allocate_buffers(
                sparse_buffers, pixels, fragment_capacity);

            cv::Mat guf_max;
            cv::Mat guf_min;
            cv::Mat ole_max;
            cv::Mat ole_min;
            cv::Mat sparse_max;
            cv::Mat sparse_min;
            Timing guf_timing;
            Timing ole_timing;
            Timing sparse_timing;
            const auto benchmark_guf = [&]() {
                guf_timing = benchmark_common_v02(
                    run_global_uf,
                    common_buffers,
                    image,
                    block,
                    warmup,
                    repetitions,
                    guf_max,
                    guf_min);
            };
            const auto benchmark_ole = [&]() {
                ole_timing = benchmark_common_v02(
                    run_flatzone_ole,
                    common_buffers,
                    image,
                    block,
                    warmup,
                    repetitions,
                    ole_max,
                    ole_min);
            };
            const auto benchmark_sparse = [&]() {
                sparse_timing = benchmark_sparse_v02(
                    sparse_buffers,
                    image,
                    warmup,
                    repetitions,
                    sparse_max,
                    sparse_min);
            };
            if (execution_order == "gos") {
                benchmark_guf();
                benchmark_ole();
                benchmark_sparse();
            } else {
                benchmark_sparse();
                benchmark_ole();
                benchmark_guf();
            }

            const std::uint64_t ole_cross_max =
                compare_masks_v02(guf_max, ole_max);
            const std::uint64_t ole_cross_min =
                compare_masks_v02(guf_min, ole_min);
            const std::uint64_t sparse_cross_max =
                compare_masks_v02(guf_max, sparse_max);
            const std::uint64_t sparse_cross_min =
                compare_masks_v02(guf_min, sparse_min);
            total_cross_mismatches +=
                ole_cross_max + ole_cross_min +
                sparse_cross_max + sparse_cross_min;

            cv::Mat expected_max;
            cv::Mat expected_min;
            if (!expected_max_dir.empty()) {
                const std::string mask_name =
                    image_path.stem().string() + ".pgm";
                expected_max = cv::imread(
                    (expected_max_dir / mask_name).string(),
                    cv::IMREAD_GRAYSCALE);
                expected_min = cv::imread(
                    (expected_min_dir / mask_name).string(),
                    cv::IMREAD_GRAYSCALE);
                if (expected_max.empty() || expected_min.empty()) {
                    throw std::runtime_error(
                        "Missing oracle masks for " +
                        image_path.filename().string());
                }
            }

            auto oracle_pair = [&](const cv::Mat& maxima,
                                   const cv::Mat& minima) {
                std::pair<std::uint64_t, std::uint64_t> result{0, 0};
                if (!expected_max.empty()) {
                    result.first =
                        compare_masks_v02(maxima, expected_max);
                    result.second =
                        compare_masks_v02(minima, expected_min);
                }
                total_oracle_mismatches +=
                    result.first + result.second;
                return result;
            };
            const auto guf_oracle = oracle_pair(guf_max, guf_min);
            const auto ole_oracle = oracle_pair(ole_max, ole_min);
            const auto sparse_oracle =
                oracle_pair(sparse_max, sparse_min);

            const Memory common_memory = common_memory_v02(pixels);
            const Memory sparse_memory =
                sparse_memory_v02(
                    pixels, fragment_capacity, tile_count);
            write_row_v02(
                csv, "Global-UF-CUDA", image_path, width, height,
                warmup, repetitions, block, common_memory, guf_timing,
                guf_oracle.first, guf_oracle.second, 0, 0);
            write_row_v02(
                csv, "FlatZone-OLE-CUDA", image_path, width, height,
                warmup, repetitions, block, common_memory, ole_timing,
                ole_oracle.first, ole_oracle.second,
                ole_cross_max, ole_cross_min);
            write_row_v02(
                csv, "Sparse-Seam-v0.9-variant1",
                image_path, width, height,
                warmup, repetitions, 256, sparse_memory, sparse_timing,
                sparse_oracle.first, sparse_oracle.second,
                sparse_cross_max, sparse_cross_min);
            csv.flush();

            std::cout
                << "[GPU-COMMON-v0.5] "
                << image_path.filename().string()
                << " GUF_ms=" << guf_timing.median_compute_ms
                << " OLE_ms=" << ole_timing.median_compute_ms
                << " SPARSE_ms=" << sparse_timing.median_compute_ms
                << " oracle_mismatch="
                << guf_oracle.first + guf_oracle.second +
                   ole_oracle.first + ole_oracle.second +
                   sparse_oracle.first + sparse_oracle.second
                << " cross_mismatch="
                << ole_cross_max + ole_cross_min +
                   sparse_cross_max + sparse_cross_min
                << '\n';

            common_free_buffers(common_buffers);
            sparse_free_buffers(sparse_buffers);
        }

        std::cout
            << "[GPU-COMMON-v0.5-OK] images=" << files.size()
            << " oracle_mismatches=" << total_oracle_mismatches
            << " cross_mismatches=" << total_cross_mismatches
            << " output=" << output_dir.string() << '\n';
        return total_oracle_mismatches == 0 &&
                       total_cross_mismatches == 0
                   ? 0
                   : 1;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 3;
    }
}
