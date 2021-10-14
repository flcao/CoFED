% create CNNs

t = [20 24 32 40 48 56 80 96];
nOut = cellfun('prodofsize', load('IIDData.mat', 'classList').classList);
nfilters = arrayfun(@( ~ )sort(t(randi(end, randi(2 : 3), 1))), 1 : n, 'un', 0);
layerList = arrayfun(@(no, nf) [helper.createLayers([32 32 3], no, nf{1}); classificationLayer('Name','classOutput')], nOut(:), nfilters(:), 'un', 0);
save layerList.mat layerList