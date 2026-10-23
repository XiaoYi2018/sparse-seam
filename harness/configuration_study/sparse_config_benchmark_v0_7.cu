// Unified in-process configuration benchmark for Sparse-Seam v0.9 variant 1.
//
// The included generated snapshot owns all device code and the unchanged
// allocate_buffers(), run_sparse_seam(), compare_masks(), and free_buffers()
// functions. This host-only wrapper adds a common correctness/timing protocol
// without reimplementing the five-kernel sequence.

#define main sparse_snapshot_disabled_main
#include "generated_sparse_seam_kernel_v0_6.cu"
#undef main

#include <cmath>
#include <cstdlib>
#include <numeric>
#include <set>

static_assert(
    SS_CLASSIFY_VARIANT == 1,
    "The preregistered ablation requires classification variant 1");
static_assert(
    (TILE == 16 && SS_CLASSIFY_THREADS == 256) ||
    (TILE == 20 && SS_CLASSIFY_THREADS == 256) ||
    (TILE == 24 && SS_CLASSIFY_THREADS == 256) ||
    (TILE == 28 && SS_CLASSIFY_THREADS == 256) ||
    (TILE == 64 && SS_CLASSIFY_THREADS == 256) ||
    (TILE == 32 && SS_CLASSIFY_THREADS == 128) ||
    (TILE == 32 && SS_CLASSIFY_THREADS == 256) ||
    (TILE == 32 && SS_CLASSIFY_THREADS == 512) ||
    (TILE == 32 && SS_CLASSIFY_THREADS == 1024),
    "The binary is outside the nine preregistered configurations");

static double percentile_config_v07(
    std::vector<double> values,
    double probability
) {
    if (values.empty()) {
        throw std::runtime_error("Cannot summarize an empty timing vector");
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

static std::string quote_csv_config_v07(const std::string& value) {
    std::string result;
    result.reserve(value.size() + 2);
    result.push_back('"');
    for (const char character : value) {
        if (character == '"') {
            result.push_back('"');
        }
        result.push_back(character);
    }
    result.push_back('"');
    return result;
}

static int parse_int_config_v07(
    const std::string& text,
    const std::string& field
) {
    std::size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(text, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid integer for " + field + ": " + text);
    }
    if (consumed != text.size()) {
        throw std::runtime_error("Invalid integer for " + field + ": " + text);
    }
    return value;
}

static std::string compiled_config_name_v07() {
    return "tile" + std::to_string(TILE) +
           "_threads" + std::to_string(SS_CLASSIFY_THREADS);
}

struct DeviceBuffersGuardV07 {
    DeviceBuffers buffers{};
    bool owns_complete_allocation = false;

    DeviceBuffersGuardV07() = default;
    DeviceBuffersGuardV07(const DeviceBuffersGuardV07&) = delete;
    DeviceBuffersGuardV07& operator=(const DeviceBuffersGuardV07&) = delete;

    void mark_allocation_complete() noexcept {
        owns_complete_allocation = true;
    }

    void release_checked() {
        if (!owns_complete_allocation) {
            return;
        }
        // Disarm before invoking the canonical release function. If a
        // cudaFree inside it reports an error, stack unwinding cannot invoke
        // the same release sequence a second time.
        owns_complete_allocation = false;
        free_buffers(buffers);
    }

    ~DeviceBuffersGuardV07() {
        if (!owns_complete_allocation) {
            return;
        }
        owns_complete_allocation = false;
        try {
            free_buffers(buffers);
        } catch (...) {
            // A destructor cannot turn a failed evidence run into OK. Any
            // exception reaching this path is already propagating to main.
        }
    }
};

static std::set<std::string> regular_filenames_v07(
    const fs::path& directory
) {
    std::set<std::string> names;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            if (entry.file_size() == 0) {
                throw std::runtime_error(
                    "Actual-mask output file is empty: " +
                    entry.path().string());
            }
            names.insert(entry.path().filename().string());
        }
    }
    return names;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr
            << "Usage: sparse_config_benchmark_v0_7.exe <image_dir>"
               " <expected_max_dir> <expected_min_dir> --output=DIR"
               " [--device=0] [--warmup=5] [--reps=10]"
               " [--round=1] [--config=NAME] [--skip-expected]\n";
        return 2;
    }

    try {
        const fs::path image_dir = argv[1];
        const fs::path expected_max_dir = argv[2];
        const fs::path expected_min_dir = argv[3];
        fs::path output_dir;
        int device = 0;
        int warmup = 5;
        int repetitions = 10;
        int round = 1;
        std::string config_name;
        bool config_provided = false;
        bool skip_expected = false;

        for (int index = 4; index < argc; ++index) {
            const std::string argument = argv[index];
            if (argument.rfind("--output=", 0) == 0) {
                output_dir = argument.substr(9);
            } else if (argument.rfind("--device=", 0) == 0) {
                device = parse_int_config_v07(
                    argument.substr(9), "--device");
            } else if (argument.rfind("--warmup=", 0) == 0) {
                warmup = parse_int_config_v07(
                    argument.substr(9), "--warmup");
            } else if (argument.rfind("--reps=", 0) == 0) {
                repetitions = parse_int_config_v07(
                    argument.substr(7), "--reps");
            } else if (argument.rfind("--round=", 0) == 0) {
                round = parse_int_config_v07(
                    argument.substr(8), "--round");
            } else if (argument.rfind("--config=", 0) == 0) {
                if (config_provided) {
                    throw std::runtime_error("Duplicate --config argument");
                }
                config_name = argument.substr(9);
                config_provided = true;
            } else if (argument == "--skip-expected") {
                skip_expected = true;
            } else {
                throw std::runtime_error("Unknown argument: " + argument);
            }
        }

        if (output_dir.empty()) {
            throw std::runtime_error("Missing --output directory");
        }
        if (fs::exists(output_dir)) {
            throw std::runtime_error(
                "Refusing to overwrite existing output directory: " +
                output_dir.string());
        }
        if (warmup < 0 || repetitions < 1 || round < 0 || round > 9) {
            throw std::runtime_error("Invalid warmup/reps/round");
        }
        const std::string compiled_config = compiled_config_name_v07();
        if (!config_provided || config_name != compiled_config) {
            throw std::runtime_error(
                "CLI config does not match compiled identity: expected " +
                compiled_config);
        }
        if (skip_expected) {
            if (round < 1 || round > 9 ||
                warmup != 5 || repetitions != 10) {
                throw std::runtime_error(
                    "Timing runs require rounds 1..9 and exact 5/10 protocol");
            }
        } else if (round != 0) {
            throw std::runtime_error(
                "Oracle admission runs must use --round=0");
        }
        if (!skip_expected &&
            (expected_max_dir.empty() || expected_min_dir.empty())) {
            throw std::runtime_error("Expected-mask directories are required");
        }

        const char* cuda_module_loading =
            std::getenv("CUDA_MODULE_LOADING");
        if (cuda_module_loading == nullptr ||
            std::string(cuda_module_loading) != "EAGER") {
            throw std::runtime_error(
                "CUDA_MODULE_LOADING must be explicitly set to EAGER");
        }

        CUDA_CHECK(cudaSetDevice(device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

        if (!fs::create_directories(output_dir)) {
            throw std::runtime_error(
                "Failed to create a new output directory");
        }
        if (!skip_expected) {
            if (!fs::create_directories(output_dir / "actual_max") ||
                !fs::create_directories(output_dir / "actual_min")) {
                throw std::runtime_error(
                    "Failed to create new actual-mask directories");
            }
        }

        std::ofstream raw_csv(output_dir / "raw_samples.csv");
        raw_csv.exceptions(
            std::ios::failbit | std::ios::badbit);
        raw_csv
            << "config,round,image,repetition,elapsed_ms\n";
        std::ofstream summary_csv(output_dir / "summary.csv");
        summary_csv.exceptions(
            std::ios::failbit | std::ios::badbit);
        summary_csv
            << "config,round,image,width,height,megapixels,warmup,"
               "repetitions,tile,classification_threads,"
               "other_kernel_threads,tiles,fragment_capacity,"
               "active_fragments,auxiliary_bytes,auxiliary_Bpp,"
               "median_compute_ms,p95_compute_ms,compute_mpix_s,"
               "oracle_checked,max_mismatches,min_mismatches\n";

        const std::vector<fs::path> files = list_images(image_dir);
        if (files.empty()) {
            throw std::runtime_error("No supported images found");
        }
        std::set<std::string> expected_actual_names;
        for (const fs::path& file : files) {
            const std::string mask_name =
                file.stem().string() + ".pgm";
            if (!expected_actual_names.insert(mask_name).second) {
                throw std::runtime_error(
                    "Input stems are not unique: " +
                    file.stem().string());
            }
        }
        if (skip_expected) {
            std::set<std::string> unique_names;
            for (const fs::path& file : files) {
                unique_names.insert(file.filename().string());
            }
            if (files.size() != 100 || unique_names.size() != 100) {
                throw std::runtime_error(
                    "Formal timing requires exactly 100 unique image files");
            }
        }

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

            DeviceBuffersGuardV07 buffer_guard;
            allocate_buffers(
                buffer_guard.buffers, pixels, fragment_capacity);
            buffer_guard.mark_allocation_complete();
            cv::Mat actual_max;
            cv::Mat actual_min;
            int active_fragments = 0;

            for (int index = 0; index < warmup; ++index) {
                (void)run_sparse_seam(
                    buffer_guard.buffers,
                    image,
                    actual_max,
                    actual_min,
                    active_fragments);
            }

            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(repetitions));
            for (int repetition = 1;
                 repetition <= repetitions;
                 ++repetition) {
                const double elapsed_ms = static_cast<double>(
                    run_sparse_seam(
                        buffer_guard.buffers,
                        image,
                        actual_max,
                        actual_min,
                        active_fragments));
                if (!(elapsed_ms > 0.0) || !std::isfinite(elapsed_ms)) {
                    throw std::runtime_error("Invalid CUDA-event time");
                }
                samples.push_back(elapsed_ms);
            }

            // Do not perform filesystem I/O between the 10 back-to-back
            // measured calls. Persist the complete sample vector only after
            // the measurement loop has ended.
            for (int repetition = 1;
                 repetition <= repetitions;
                 ++repetition) {
                raw_csv
                    << quote_csv_config_v07(config_name) << ','
                    << round << ','
                    << quote_csv_config_v07(
                           image_path.filename().string()) << ','
                    << repetition << ','
                    << std::fixed << std::setprecision(9)
                    << samples[
                           static_cast<std::size_t>(repetition - 1)]
                    << '\n';
            }

            const std::uint64_t max_mismatches =
                skip_expected ? 0 : compare_masks(actual_max, expected_max);
            const std::uint64_t min_mismatches =
                skip_expected ? 0 : compare_masks(actual_min, expected_min);
            total_max_mismatches += max_mismatches;
            total_min_mismatches += min_mismatches;
            total_pixels += static_cast<std::uint64_t>(pixels);

            const double median_ms =
                percentile_config_v07(samples, 0.5);
            const double p95_ms =
                percentile_config_v07(samples, 0.95);
            const double megapixels =
                static_cast<double>(pixels) / 1.0e6;
            const std::uint64_t auxiliary_bytes =
                static_cast<std::uint64_t>(pixels) +
                static_cast<std::uint64_t>(fragment_capacity) * 16ull +
                static_cast<std::uint64_t>(tiles) *
                    sizeof(std::uint16_t);

            summary_csv
                << quote_csv_config_v07(config_name) << ','
                << round << ','
                << quote_csv_config_v07(
                       image_path.filename().string()) << ','
                << width << ',' << height << ','
                << std::fixed << std::setprecision(6)
                << megapixels << ','
                << warmup << ',' << repetitions << ','
                << TILE << ',' << SS_CLASSIFY_THREADS << ",256,"
                << tiles << ',' << fragment_capacity << ','
                << active_fragments << ',' << auxiliary_bytes << ','
                << std::fixed << std::setprecision(6)
                << static_cast<double>(auxiliary_bytes) /
                       static_cast<double>(pixels) << ','
                << std::fixed << std::setprecision(9)
                << median_ms << ',' << p95_ms << ','
                << (1000.0 * megapixels / median_ms) << ','
                << (skip_expected ? 0 : 1) << ','
                << max_mismatches << ','
                << min_mismatches << '\n';
            if (!skip_expected) {
                const bool max_written = cv::imwrite(
                    (output_dir / "actual_max" /
                     (image_path.stem().string() + ".pgm")).string(),
                    actual_max);
                const bool min_written = cv::imwrite(
                    (output_dir / "actual_min" /
                     (image_path.stem().string() + ".pgm")).string(),
                    actual_min);
                if (!max_written || !min_written) {
                    throw std::runtime_error(
                        "Failed to write actual mask images for " +
                        image_path.filename().string());
                }
            }

            buffer_guard.release_checked();
            std::cout
                << "[SPARSE-CONFIG-v0.7] "
                << config_name << " round=" << round << ' '
                << image_path.filename().string()
                << " median_ms=" << median_ms
                << " p95_ms=" << p95_ms
                << " active_fragments=" << active_fragments
                << " mismatch="
                << max_mismatches + min_mismatches << '\n';
        }

        raw_csv.close();
        summary_csv.close();

        if (!skip_expected) {
            const std::set<std::string> actual_max_names =
                regular_filenames_v07(output_dir / "actual_max");
            const std::set<std::string> actual_min_names =
                regular_filenames_v07(output_dir / "actual_min");
            if (actual_max_names != expected_actual_names ||
                actual_min_names != expected_actual_names) {
                throw std::runtime_error(
                    "Actual-mask output name set is incomplete or unexpected");
            }
        }

        std::ofstream environment(output_dir / "environment.txt");
        environment.exceptions(
            std::ios::failbit | std::ios::badbit);
        environment
            << "algorithm=Sparse-Seam-v0.9-variant1\n"
            << "harness=sparse_config_benchmark_v0.7_inprocess\n"
            << "config=" << config_name << '\n'
            << "round=" << round << '\n'
            << "device_name=" << properties.name << '\n'
            << "compute_capability=" << properties.major << '.'
            << properties.minor << '\n'
            << "tile=" << TILE << '\n'
            << "classification_threads=" << SS_CLASSIFY_THREADS << '\n'
            << "other_kernel_threads=256\n"
            << "fragment_slots_per_tile=" << SLOTS_PER_TILE << '\n'
            << "classification_variant=" << SS_CLASSIFY_VARIANT << '\n'
            << "CUDA_MODULE_LOADING=" << cuda_module_loading << '\n'
            << "warmup=" << warmup << '\n'
            << "repetitions=" << repetitions << '\n'
            << "raw_samples_saved=true\n"
            << "timing=cuda_event_five_kernel_sequence\n"
            << "event_create_destroy_inside_run_outside_event_interval=true\n"
            << "oracle_checked=" << (skip_expected ? 0 : 1) << '\n';
        environment.close();

        if (total_max_mismatches != 0 || total_min_mismatches != 0) {
            throw std::runtime_error("Pixelwise validation failed");
        }

        std::cout
            << "[SPARSE-CONFIG-v0.7-OK] config=" << config_name
            << " round=" << round
            << " images=" << files.size()
            << " pixels=" << total_pixels
            << " oracle_checked=" << (skip_expected ? 0 : 1)
            << " raw_samples="
            << static_cast<std::uint64_t>(files.size()) *
                   static_cast<std::uint64_t>(repetitions)
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[SPARSE-CONFIG-v0.7-FAIL] "
                  << error.what() << '\n';
        return 1;
    }
}
