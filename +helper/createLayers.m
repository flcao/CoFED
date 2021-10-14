function layers = createLayers(inSize, outSize, nFilters)
  
  layers = [
    imageInputLayer(inSize, "Normalization", "none", "Name", "input")
    
    helper(nFilters(1), 1)
    maxPooling2dLayer(2, "Stride", 2, "Name", "maxpool_1")
    
    helper(nFilters(2), 2)
    maxPooling2dLayer(2, "Stride", 2, "Name", "maxpool_2")
    ];
  
  if numel(nFilters) == 3, layers = [layers; helper(nFilters(3), 3)]; end
  
  layers = [layers
    globalAveragePooling2dLayer("Name", "gapool")
    fullyConnectedLayer(outSize, "Name", "dense", "BiasLearnRateFactor", 0)
    softmaxLayer("Name", "softmax")
%     classificationLayer("Name", "classout")
    ];
  
function layers = helper(n, k)
  layers = [
    convolution2dLayer(3, n(1), "Name", "conv_" + k, "BiasInitializer", "narrow-normal")
    batchNormalizationLayer("Name", "batchnorm_" + k, ...
    "OffsetInitializer", "narrow-normal", "OffsetL2Factor", 0, "ScaleL2Factor", 0)
    reluLayer("Name", "relu_" + k)];