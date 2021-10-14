% create femnist dataset (100 writers)

data = load('femnist.mat');
im = single(data.X) * 2 / 255 - 1;
c = 1 : 62;
cc = categorical(string(data.y));
[writers, ~, idx] = unique(data.writers);
idx = accumarray(idx, (1 : numel(idx))', [], @cellhorzcat);
classList = repmat({c'}, numel(writers), 1);
trainList = cell(numel(writers), 1);
testList = trainList;
r = 0.4;
rng(7)
for i = 1 : numel(idx)
    ii = idx{i};
    t = ii(randperm(end, round(end * r)));
    trainList{i} = {im(:, :, :, t), cc(t)};
    t = setdiff(ii, t);
    testList{i} = {im(:, :, :, t), cc(t)};
    
end
save IIDData_FE classList trainList testList writers