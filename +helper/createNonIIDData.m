function [trainList, fineClassList] = createNonIIDData(coarseClassList, xData, yDataCoarse, yDataFine, nPerClass, nlim)
    
    fineOfCoarse = accumarray(yDataCoarse, yDataFine, [], @(x){unique(x)});
    coarse = vertcat(coarseClassList{:});
    fine = zeros(size(coarse));
    for k = unique(coarse)'
        while true
            idx = coarse == k;
            t = fineOfCoarse{k}(randi(end, nnz(idx), 1));
            f = histcounts(t, [unique(t); inf]);
            if numel(f) == 5 && max(f) - min(f) <= 4 && max(f) * nPerClass <= nlim, break, end
        end
        fine(idx) = t;
    end
    fineClassList = mat2cell(fine, cellfun('prodofsize', coarseClassList));
    trainList = helper.createIIDData(fineClassList, xData, yDataFine, nPerClass, 0);