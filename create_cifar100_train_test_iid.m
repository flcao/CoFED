% clear CIFAR100 IID train set (100 clients)

% https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz

t = 1 : 20;
classList = arrayfun(@( ~ )sort(t(randperm(20, randi([6 8], 1))))', 1 : n, 'un', 0)';

xdata = load('CIFAR.mat', 'cf100TrainImage').cf100TrainImage;
xdata = 2 * single(xdata) / 255 - 1;
ydata = load('CIFAR.mat', 'cf100TrainCoarseLabel').cf100TrainCoarseLabel + 1;
trainList = helper.createIIDData(classList, xdata, ydata, 50, 0);

xdata = load('CIFAR.mat', 'cf100TestImage').cf100TestImage;
xdata = 2 * single(xdata) / 255 - 1;
ydata = load('CIFAR.mat', 'cf100TestCoarseLabel').cf100TestCoarseLabel + 1;
testList = helper.createIIDData(classList, xdata, ydata, 500, 1);
% save IIDData classList trainList testList -v7.3 -nocompression