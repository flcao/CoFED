clear, clc

import helper.*

data = load('nonIIDData.mat');
classList = data.classList;
trainList = data.trainList;
testList = data.testList;
xPublic = 2 * single(load('CIFAR.mat', 'cf10TrainImage').cf10TrainImage) / 255 - 1;
layerList = load('layerList.mat').layerList;
layerList = cellfun(@(l)[layerList{1}(1 : end - 3); l(end - 2 : end)], layerList, 'un', 0);


netList = load("bestIIDloc.mat", "bestModel").bestModel;
bestOption = load("bestIIDloc.mat", "bestOption").bestOption;
bestAcc = load("bestIIDloc.mat", "bestAcc").bestAcc;

import customLayers.weightedClassificationLayer
fedNetList = cell(size(netList));
fedAccuracy = zeros(size(netList));
idxList = fedLabelling(netList, classList, xPublic, repelem(1 / numel(netList), numel(netList), 1), .3);
bestFedOption = cell(size(netList));

for i = 1 : numel(fedNetList)
  
  classIdx = removeOverlap(idxList(classList{i}));
  xTrain = cat(4, trainList{i}{1}, xPublic(:, :, :, vertcat(classIdx{:})));
  yTrain = [trainList{i}{2}; repelem(categorical(1 : numel(classList{i}))', cellfun('prodofsize', classIdx))];
  
  classWeights = 1 ./ countcats(yTrain);
  %   classWeights(isinf(classWeights)) = 0;
  outputlayer = weightedClassificationLayer(classWeights / max(classWeights), netList{i}.OutputNames{1});
  layer = replaceOutputLayer(netList{i}, outputlayer);
  
  option = bestOption{i};
  option.MiniBatchSize = 1000;
  option.MaxEpochs = 200;
  option.ValidationFrequency = ceil(numel(yTrain) / option.MiniBatchSize);
  option.OutputFcn = @(info)isstop(info, 10, .1);
  for lr = [.1 2] * option.InitialLearnRate
    option.InitialLearnRate = lr;
    f = trainNetwork(xTrain, yTrain, layer, option);
    acc = mean(f.classify(testList{i}{1}) == testList{i}{2});
    if acc > fedAccuracy(i)
      fedAccuracy(i) = acc;
      fedNetList{i} = f;
      bestFedOption{i} = option;
    end
  end
  
  if mod(i, 5) == 0, save(sprintf('temp%d', i), 'fedAccuracy', 'fedNetList', 'bestFedOption'), end
end
save bestIIDfed_cifar10pub fedAccuracy fedNetList bestFedOption

function idxList = fedLabelling(netList, classList, x, netWeights, threshold)
  [y, k] = cellfun(@(net) maxk(net.predict(x), 2, 2), netList, 'un', 0);
  idx = cellfun(@(y) find(diff(y, 1, 2) < -0.3), y, 'un', 0);
  class = cellfun(@(c, k, i) c(k(i, 1)), classList, k, idx, 'un', 0);
  idxList = accumarray(vertcat(class{:}), vertcat(idx{:}), [], @cellhorzcat);
  weightList = accumarray(vertcat(class{:}), repelem(netWeights(:), cellfun('prodofsize', class)), [], @cellhorzcat);
  
  total = accumarray(vertcat(classList{:}), repelem(netWeights(:), cellfun('prodofsize', classList)));
  
  for i = find(cellfun('prodofsize', idxList))'
    [u, ~, iu] = unique(idxList{i});
    idxList{i} = u(accumarray(iu(:), weightList{i}) / total(i) > threshold);
  end
end

function list = removeOverlap(list)
  a = vertcat(list{:});
  [u, ~, iu] = unique(a);
  c = histcounts(a, [u; inf])';
  a = a .* (c(iu) == 1);
  list = cellfun(@nonzeros, mat2cell(a, cellfun('prodofsize', list)), 'un', 0);
end
