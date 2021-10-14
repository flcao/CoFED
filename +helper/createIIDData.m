function dataList = createIIDData(classList, xData, yData, nPerClass, overlapFlag)
  
  if nargin < 5 || ~overlapFlag
    
    classIdx = accumarray(yData, (1 : numel(yData))', [], @cellhorzcat);
    nOwners = numel(classList);
    dataList = cell(nOwners, 1);
    
    for i = 1 : nOwners
      idx = [];
      for j = classList{i}'
        p = randperm(numel(classIdx{j}), nPerClass);
        idx = [idx; classIdx{j}(p)]; %#ok<*AGROW>
        classIdx{j}(p) = [];
      end
      x = xData(:, :, :, idx);
      y = repelem(categorical(1 : numel(classList{i}))', nPerClass);
      dataList{i} = {x, y};
    end
    
  else
    
    classIdx = accumarray(yData, (1 : numel(yData))', [], @cellhorzcat);
    idx = cellfun(@(class) classIdx(class), classList, 'un', 0);
    x = cellfun(@(idx) xData(:, :, :, vertcat(idx{:})), idx, 'un', 0);
    y = cellfun(@(idx) repelem(categorical(1 : numel(idx))', nPerClass), idx, 'un', 0);
    dataList = cellfun(@cellhorzcat, x, y, 'un', 0);
    
  end