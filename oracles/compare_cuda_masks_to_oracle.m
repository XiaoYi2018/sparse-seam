function compare_cuda_masks_to_oracle(oracleRoot, actualDir)
% Compare *_max.png and *_min.png masks with MATLAB oracle PGM masks.

maxDir = fullfile(oracleRoot, 'expected_max');
minDir = fullfile(oracleRoot, 'expected_min');
expected = dir(fullfile(maxDir, '*.pgm'));
n = numel(expected);

caseIds = strings(n, 1);
maxMismatches = zeros(n, 1);
minMismatches = zeros(n, 1);
statuses = strings(n, 1);

for idx = 1:n
    [~, caseId, ~] = fileparts(expected(idx).name);
    expectedMax = imread(fullfile(maxDir, [caseId, '.pgm'])) ~= 0;
    expectedMin = imread(fullfile(minDir, [caseId, '.pgm'])) ~= 0;
    actualMaxPath = fullfile(actualDir, [caseId, '_max.png']);
    actualMinPath = fullfile(actualDir, [caseId, '_min.png']);
    assert(exist(actualMaxPath, 'file') == 2, ...
        'Missing actual maximum mask: %s', actualMaxPath);
    assert(exist(actualMinPath, 'file') == 2, ...
        'Missing actual minimum mask: %s', actualMinPath);
    actualMax = imread(actualMaxPath) ~= 0;
    actualMin = imread(actualMinPath) ~= 0;
    assert(isequal(size(actualMax), size(expectedMax)), ...
        'Maximum mask size mismatch for %s', caseId);
    assert(isequal(size(actualMin), size(expectedMin)), ...
        'Minimum mask size mismatch for %s', caseId);

    caseIds(idx) = string(caseId);
    maxMismatches(idx) = nnz(actualMax ~= expectedMax);
    minMismatches(idx) = nnz(actualMin ~= expectedMin);
    if maxMismatches(idx) == 0 && minMismatches(idx) == 0
        statuses(idx) = "PASS";
    else
        statuses(idx) = "FAIL";
    end
end

audit = table( ...
    caseIds, maxMismatches, minMismatches, statuses, ...
    'VariableNames', { ...
        'case_id', 'max_mismatches', 'min_mismatches', 'status' ...
    } ...
);
writetable(audit, fullfile(actualDir, 'pixelwise_oracle_audit.csv'));

totalMax = sum(maxMismatches);
totalMin = sum(minMismatches);
passed = nnz(statuses == "PASS");
fprintf( ...
    '[CUDA-PIXELWISE-AUDIT] cases=%d passed=%d max_mismatches=%d min_mismatches=%d\n', ...
    n, passed, totalMax, totalMin ...
);
assert(passed == n && totalMax == 0 && totalMin == 0, ...
    'CUDA output differs from MATLAB oracle.');
end
