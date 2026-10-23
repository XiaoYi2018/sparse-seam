function build_fuzz_oracle_v0_2
% Build a deterministic fuzz corpus for exact 8-connected regional extrema.
% MATLAB imregionalmax/imregionalmin remain the external semantic oracle.

scriptDir = fileparts(mfilename('fullpath'));
outDir = fullfile(scriptDir, 'oracle_v0_2_fuzz');
inputDir = fullfile(outDir, 'inputs');
inputPngDir = fullfile(outDir, 'inputs_png');
maxDir = fullfile(outDir, 'expected_max');
minDir = fullfile(outDir, 'expected_min');
dirs = {outDir, inputDir, inputPngDir, maxDir, minDir};
for idx = 1:numel(dirs)
    if ~exist(dirs{idx}, 'dir')
        mkdir(dirs{idx});
    end
end

rng(20260728, 'twister');
sizes = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 97];
levels = [2, 3, 4, 8, 16, 32, 256];
caseCount = 160;

ids = strings(caseCount, 1);
families = strings(caseCount, 1);
heights = zeros(caseCount, 1);
widths = zeros(caseCount, 1);
quantLevels = zeros(caseCount, 1);
maxPixels = zeros(caseCount, 1);
minPixels = zeros(caseCount, 1);
maxComponents = zeros(caseCount, 1);
minComponents = zeros(caseCount, 1);
extremeUnionPixels = zeros(caseCount, 1);
extremePlateauCount = zeros(caseCount, 1);

for idx = 1:caseCount
    if mod(idx, 16) == 1
        h = 1;
        w = sizes(mod(idx * 5 - 1, numel(sizes)) + 1);
    elseif mod(idx, 16) == 2
        h = sizes(mod(idx * 7 - 1, numel(sizes)) + 1);
        w = 1;
    else
        h = sizes(randi(numel(sizes)));
        w = sizes(randi(numel(sizes)));
    end
    q = levels(mod(idx * 3 - 1, numel(levels)) + 1);

    familyIndex = mod(idx - 1, 4);
    if familyIndex == 0
        family = "iid";
        img = uint8(randi([0, q - 1], h, w));
    elseif familyIndex == 1
        family = "block";
        blockH = 1 + mod(idx * 5, 8);
        blockW = 1 + mod(idx * 7, 8);
        seedH = ceil(h / blockH);
        seedW = ceil(w / blockW);
        seed = uint8(randi([0, q - 1], seedH, seedW));
        img = kron(seed, ones(blockH, blockW, 'uint8'));
        img = img(1:h, 1:w);
    elseif familyIndex == 2
        family = "mixed";
        img = uint8(randi([0, q - 1], h, w));
        if h >= 3 && w >= 3
            y0 = 1 + mod(idx * 3, h);
            x0 = 1 + mod(idx * 5, w);
            y1 = min(h, y0 + mod(idx, min(12, h)));
            x1 = min(w, x0 + mod(idx * 2, min(12, w)));
            img(y0:y1, x0:x1) = uint8(mod(idx * 11, q));
        end
    else
        family = "quantized_gradient";
        [yy, xx] = ndgrid(0:h-1, 0:w-1);
        noise = randi([0, max(1, floor(q / 8))], h, w);
        img = uint8(mod(3 * xx + 5 * yy + noise, q));
    end

    maxMask = imregionalmax(img, 8);
    minMask = imregionalmin(img, 8);
    maxCC = bwconncomp(maxMask, 8);
    minCC = bwconncomp(minMask, 8);
    overlapCC = bwconncomp(maxMask & minMask, 8);

    caseId = sprintf('fuzz_%03d_%s_%dx%d_q%d', idx, char(family), h, w, q);
    ids(idx) = string(caseId);
    families(idx) = family;
    heights(idx) = h;
    widths(idx) = w;
    quantLevels(idx) = q;
    maxPixels(idx) = nnz(maxMask);
    minPixels(idx) = nnz(minMask);
    maxComponents(idx) = maxCC.NumObjects;
    minComponents(idx) = minCC.NumObjects;
    extremeUnionPixels(idx) = nnz(maxMask | minMask);
    extremePlateauCount(idx) = ...
        maxCC.NumObjects + minCC.NumObjects - overlapCC.NumObjects;

    imwrite(img, fullfile(inputDir, [caseId, '.pgm']));
    imwrite(img, fullfile(inputPngDir, [caseId, '.png']));
    imwrite(uint8(maxMask) * 255, fullfile(maxDir, [caseId, '.pgm']));
    imwrite(uint8(minMask) * 255, fullfile(minDir, [caseId, '.pgm']));
end

manifest = table( ...
    ids, families, heights, widths, quantLevels, ...
    maxPixels, minPixels, maxComponents, minComponents, ...
    extremeUnionPixels, extremePlateauCount, ...
    'VariableNames', { ...
        'case_id', 'family', 'height', 'width', 'quant_levels', ...
        'max_pixels', 'min_pixels', 'max_components_8c', ...
        'min_components_8c', 'extreme_union_pixels', ...
        'extreme_plateau_count_union' ...
    } ...
);
writetable(manifest, fullfile(outDir, 'oracle_manifest.csv'));

readmePath = fullfile(outDir, 'README.txt');
fid = fopen(readmePath, 'w');
assert(fid >= 0, 'Could not open README for writing.');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'Deterministic fuzz oracle v0.2\n');
fprintf(fid, 'Generated with MATLAB %s.\n', version);
fprintf(fid, 'Reference: imregionalmax(image, 8) and imregionalmin(image, 8).\n');
fprintf(fid, 'Seed: 20260728; cases: %d.\n', caseCount);
fprintf(fid, 'Families: iid, block, mixed, quantized_gradient.\n');
fprintf(fid, 'Dimensions include 1D and boundaries around common tile sizes.\n');
fprintf(fid, 'Quantization levels: 2, 3, 4, 8, 16, 32, 256.\n');

fprintf('[FUZZ-ORACLE-BUILD-OK] cases=%d output=%s\n', caseCount, outDir);
end
