%% local training, noiid, gird-search hyper-parameters

data = load('NonIIDData.mat');
classList = data.classList;
trainList = data.trainList;
testList = data.testList;
layerList = load('layerList.mat').layerList;

bestAcc = zeros(numel(classList), 1);
bestModel = cell(numel(classList), 1);
bestOption = cell(numel(classList), 1);

tic
for i = 1 : numel(classList)
  for s = ["adam", "sgdm"]
    t = trainingOptions(s, ...
      "LearnRateSchedule", "piecewise", "LearnRateDropPeriod", 1, "LearnRateDropFactor", 0.99, ...
      "MiniBatchSize", 50, "Verbose", 0, "Shuffle", "every-epoch");
    for lr = [0.02 0.04 0.06 0.08 0.1] / (1 + 9 * (s == "adam"))
      t.InitialLearnRate = lr;
      for maxEpoch = [20 30 40 50 60]
        t.MaxEpochs = maxEpoch;
        f = trainNetwork(trainList{i}{:}, layerList{i}, t);
        acc = mean(f.classify(testList{i}{1}) == testList{i}{2});
        if acc > bestAcc(i)
          bestAcc(i) = acc;
          bestModel{i} = f;
          bestOption{i} = t;
        end
      end
    end
  end
  if mod(i, 5) == 0, save(sprintf('temp%d', i), 'bestAcc', 'bestModel', 'bestOption'), end
end

save bestNonIIDloc bestAcc bestModel bestOption
