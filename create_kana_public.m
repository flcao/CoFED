% Chars74K-Kannada public dataset

f = dir('kan\**\*.png');
p = string(fullfile({f.folder}, {f.name}));
im = zeros(32, 32, 1, numel(p) * 3, 'uint8');
for i = 1 : numel(p)
    t = imread(p(i));
    t = imresize(t(:, :, 1), 64 / size(t, 1));
    im(:, :, 1, 3 * (i - 1) + 1) = imcrop(t, randomCropWindow2d(size(t), [32 32]));
    im(:, :, 1, 3 * (i - 1) + 2) = imcrop(t, randomCropWindow2d(size(t), [32 32]));
    im(:, :, 1, 3 * (i - 1) + 3) = imcrop(t, randomCropWindow2d(size(t), [32 32]));
end
save +data/ken im