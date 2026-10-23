#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::uint32_t mix32(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

static std::uint8_t quantized_level(
    std::uint32_t value,
    int levels) {
    const int bin = static_cast<int>(value % levels);
    return static_cast<std::uint8_t>(
        (bin * 255 + (levels - 1) / 2) /
        (levels - 1));
}

static void write_image(
    const fs::path& path,
    int size,
    const std::string& pattern) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error(
            "Cannot create " + path.string());
    }
    output << "P5\n" << size << ' ' << size << "\n255\n";
    std::vector<std::uint8_t> row(
        static_cast<std::size_t>(size));
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (pattern == "iid_q256") {
                const std::uint32_t key =
                    static_cast<std::uint32_t>(x) *
                        0x9e3779b9u ^
                    static_cast<std::uint32_t>(y) *
                        0x85ebca6bu ^
                    static_cast<std::uint32_t>(size);
                row[static_cast<std::size_t>(x)] =
                    static_cast<std::uint8_t>(mix32(key) & 0xffu);
            } else if (pattern == "blocks_q8") {
                const std::uint32_t block_x =
                    static_cast<std::uint32_t>(x / 32);
                const std::uint32_t block_y =
                    static_cast<std::uint32_t>(y / 32);
                const std::uint32_t key =
                    block_x * 0x9e3779b9u ^
                    block_y * 0x85ebca6bu ^
                    static_cast<std::uint32_t>(size);
                row[static_cast<std::size_t>(x)] =
                    quantized_level(mix32(key), 8);
            } else if (pattern == "diagonal_q32") {
                const int period = size >= 1024 ? size / 32 : 32;
                const int band =
                    ((x + y) / period) % 32;
                row[static_cast<std::size_t>(x)] =
                    quantized_level(
                        static_cast<std::uint32_t>(band), 32);
            } else {
                throw std::runtime_error(
                    "Unknown pattern " + pattern);
            }
        }
        output.write(
            reinterpret_cast<const char*>(row.data()),
            static_cast<std::streamsize>(row.size()));
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr
            << "Usage: generate_scaling_suite_v0_1.exe OUTPUT_DIR\n";
        return 2;
    }
    try {
        const fs::path output_dir = argv[1];
        fs::create_directories(output_dir);
        const std::vector<int> sizes{
            256, 512, 1024, 2048, 4096, 8192};
        const std::vector<std::string> patterns{
            "iid_q256", "blocks_q8", "diagonal_q32"};
        std::ofstream manifest(output_dir / "manifest.csv");
        manifest << "file,width,height,pattern,seed_rule\n";
        for (const int size : sizes) {
            for (const std::string& pattern : patterns) {
                const std::string filename =
                    pattern + "_" + std::to_string(size) +
                    "x" + std::to_string(size) + ".pgm";
                write_image(output_dir / filename, size, pattern);
                manifest
                    << filename << ',' << size << ',' << size << ','
                    << pattern
                    << ",deterministic_xy_hash_or_fixed_diagonal\n";
                std::cout
                    << "[SCALING-GENERATOR] " << filename << '\n';
            }
        }
        std::cout
            << "[SCALING-GENERATOR-v0.1-OK] images="
            << sizes.size() * patterns.size()
            << " output=" << output_dir.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 3;
    }
}
