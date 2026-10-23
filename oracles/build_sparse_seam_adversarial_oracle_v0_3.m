function build_sparse_seam_adversarial_oracle_v0_3
% Adversarial MATLAB oracle for the sparse-seam CUDA kernels.
% These cases deliberately cross 64x64 tile seams and tile corners.

scriptDir = fileparts(mfilename('fullpath'));
outDir = fullfile(scriptDir, 'oracle_v0_3_sparse_seam_adversarial');
inputDir = fullfile(outDir, 'inputs');
maxDir = fullfile(outDir, 'expected_max');
minDir = fullfile(outDir, 'expected_min');
dirs = {outDir, inputDir, maxDir, minDir};
for idx = 1:numel(dirs)
    if ~exist(dirs{idx}, 'dir')
        mkdir(dirs{idx});
    end
end

rng(20260728, 'twister');
images = cell(12, 1);
names = strings(12, 1);

images{1} = uint8(ones(128, 128) * 77);
names(1) = "constant_128x128";

images{2} = uint8(ones(129, 257) * 193);
names(2) = "constant_129x257";

img = uint8(ones(129, 257) * 20);
img(60:70, 4:253) = 180;
img(63:67, 1:257) = 180;
images{3} = img;
names(3) = "horizontal_plateau_all_vertical_seams_129x257";

img = uint8(ones(257, 129) * 220);
img(4:253, 60:70) = 40;
img(1:257, 63:67) = 40;
images{4} = img;
names(4) = "vertical_plateau_all_horizontal_seams_257x129";

img = uint8(ones(193, 193) * 20);
for idx = 1:193
    img(idx, idx) = 240;
    if idx < 193
        img(idx, idx + 1) = 240;
    end
end
images{5} = img;
names(5) = "diagonal_corner_chain_193x193";

[yy, xx] = ndgrid(0:128, 0:128);
images{6} = uint8(mod(xx + yy, 2) * 255);
names(6) = "checkerboard_129x129";

images{7} = uint8(mod(floor(xx / 3) + floor(yy / 5), 4) * 60);
names(7) = "quantized_stripes_129x129";

[yy, xx] = ndgrid(1:257, 1:257);
radius = floor(sqrt((xx - 129).^2 + (yy - 129).^2));
images{8} = uint8(mod(floor(radius / 7), 8) * 31);
names(8) = "nested_rings_257x257";

seed = uint8(randi([0, 3], 25, 33) * 70);
img = kron(seed, ones(8, 8, 'uint8'));
images{9} = img(1:193, 1:257);
names(9) = "random_blocks_q4_193x257";

[yy, xx] = ndgrid(0:192, 0:256);
images{10} = uint8(mod(3 * xx + 5 * yy, 16) * 16);
names(10) = "quantized_gradient_193x257";

img = uint8(ones(193, 257) * 100);
img(1, 1) = 101;
images{11} = img;
names(11) = "giant_plateau_single_greater_193x257";

img = uint8(ones(193, 257) * 100);
img(193, 257) = 99;
images{12} = img;
names(12) = "giant_plateau_single_smaller_193x257";

heights = zeros(12, 1);
widths = zeros(12, 1);
maxPixels = zeros(12, 1);
minPixels = zeros(12, 1);
for idx = 1:numel(images)
    img = images{idx};
    maxMask = imregionalmax(img, 8);
    minMask = imregionalmin(img, 8);
    heights(idx) = size(img, 1);
    widths(idx) = size(img, 2);
    maxPixels(idx) = nnz(maxMask);
    minPixels(idx) = nnz(minMask);
    imwrite(img, fullfile(inputDir, names(idx) + ".pgm"));
    imwrite(uint8(maxMask) * 255, fullfile(maxDir, names(idx) + ".pgm"));
    imwrite(uint8(minMask) * 255, fullfile(minDir, names(idx) + ".pgm"));
end

manifest = table( ...
    names, heights, widths, maxPixels, minPixels, ...
    'VariableNames', {'case_id', 'height', 'width', 'max_pixels', 'min_pixels'});
writetable(manifest, fullfile(outDir, 'oracle_manifest.csv'));

fid = fopen(fullfile(outDir, 'README.txt'), 'w');
assert(fid >= 0, 'Could not open README.');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'Sparse-seam adversarial oracle v0.3\n');
fprintf(fid, 'MATLAB %s; 8-connectivity.\n', version);
fprintf(fid, 'Reference: imregionalmax(image, 8), imregionalmin(image, 8).\n');
fprintf(fid, 'Tile size attacked: 64; cases: %d.\n', numel(images));
fprintf('[SPARSE-SEAM-ADVERSARIAL-ORACLE-OK] cases=%d output=%s\n', ...
    numel(images), outDir);
end
