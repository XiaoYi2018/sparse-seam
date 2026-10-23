// Scaling-suite generator v0.2.
//
// v0.1 emitted "blocks_q8" with a 32x32 block grid whose origin coincides with
// the 32x32 tile grid of the evaluated configuration, so every block plateau is
// tile-contained by construction.  v0.2 keeps that pattern unchanged as an
// in-campaign control and adds "blocks_q8_offset", which shifts the same block
// grid by half a tile in both axes.  Every interior block then straddles a 2x2
// group of tiles, which is the worst alignment for a tile-local method.
//
// v0.1 is frozen and is not modified by this file.

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

// Identical to v0.1 so that the aligned control reproduces the original
// blocks_q8 images bit for bit.
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

    // Half-tile shift for the offset variant.  The block grid of the aligned
    // variant starts at 0; adding 16 before the division moves every block
    // boundary to the middle of a tile.
    const int shift = (pattern == "blocks_q8_offset") ? 16 : 0;

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            if (pattern == "blocks_q8" ||
                pattern == "blocks_q8_offset") {
                const std::uint32_t block_x =
                    static_cast<std::uint32_t>((x + shift) / 32);
                const std::uint32_t block_y =
                    static_cast<std::uint32_t>((y + shift) / 32);
                const std::uint32_t key =
                    block_x * 0x9e3779b9u ^
                    block_y * 0x85ebca6bu ^
                    static_cast<std::uint32_t>(size);
                row[static_cast<std::size_t>(x)] =
                    quantized_level(mix32(key), 8);
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
            << "Usage: generate_scaling_suite_v0_2.exe OUTPUT_DIR\n";
        return 2;
    }
    try {
        const fs::path output_dir = argv[1];
        fs::create_directories(output_dir);
        const std::vector<int> sizes{
            256, 512, 1024, 2048, 4096, 8192};
        const std::vector<std::string> patterns{
            "blocks_q8", "blocks_q8_offset"};
        std::ofstream manifest(output_dir / "manifest.csv");
        manifest << "file,width,height,pattern,block_origin_shift\n";
        for (const int size : sizes) {
            for (const std::string& pattern : patterns) {
                const std::string filename =
                    pattern + "_" + std::to_string(size) +
                    "x" + std::to_string(size) + ".pgm";
                write_image(output_dir / filename, size, pattern);
                manifest
                    << filename << ',' << size << ',' << size << ','
                    << pattern << ','
                    << ((pattern == "blocks_q8_offset") ? 16 : 0)
                    << '\n';
                std::cout
                    << "[SCALING-GENERATOR-v0.2] " << filename << '\n';
            }
        }
        std::cout
            << "[SCALING-GENERATOR-v0.2-OK] images="
            << sizes.size() * patterns.size()
            << " output=" << output_dir.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << '\n';
        return 3;
    }
}
