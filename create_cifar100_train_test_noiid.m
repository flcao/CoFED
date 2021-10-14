% clear CIFAR100 non-IID train set (100 clients)

% https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz

classList = load('IIDData.mat', 'classList').classList; % same coarselabel as iid

xdata = load('CIFAR.mat', 'cf100TrainImage').cf100TrainImage;
xdata = 2 * single(xdata) / 255 - 1;
ydataC = load('CIFAR.mat', 'cf100TrainCoarseLabel').cf100TrainCoarseLabel + 1;
ydataF = load('CIFAR.mat', 'cf100TrainFineLabel').cf100TrainFineLabel + 1;

trainList = helper.createNonIIDData(classList, xdata, ydataC, ydataF, 50, 500);
testList = load('IIDData.mat', 'testList').testList;
% save NonIIDData classList trainList testList -v7.3 -nocompression