% 4 different classifiers, 100 clients, census1994

%% dataset
data = load('census1994.mat');
x_train_all = data.adultdata(:, 1 : end - 1);
y_train_all = data.adultdata{:, "salary"};
x_test = data.adulttest(:, 1 : end - 1);
y_test = data.adulttest{:, "salary"};

%% classifiers & train methods
n_classifier_each = 25;
fit_methods = repelem({@(x, y) fitctree(x, y, 'PredictorSelection', 'curvature', 'Surrogate', 'on')
  @(x, y) fitcsvm(x, y, 'Standardize', true, 'KernelFunction', 'RBF', 'KernelScale', 'auto')
  @(x, y) fitcgam(x, y)
  @(x, y)fitcnet(x, y, "Standardize", true)}', n_classifier_each);

%% dataset split
n_client = numel(fit_methods);
n_train = 200;
idx = randperm(height(x_train_all), n_train * n_client);
idx = mat2cell(idx(1 : n_train * n_client), 1, n_train + zeros(1, n_client));
x_train = cellfun(@(i) x_train_all(i, :), idx, 'UniformOutput', 0);
y_train = cellfun(@(i) y_train_all(i, :), idx, 'UniformOutput', 0);

%% public dataset
n_public = 5e3;
x_public = varfun(@(v)v(randperm(end, n_public)), x_train_all);
x_public = renamevars(x_public, x_public.Properties.VariableNames, x_train_all.Properties.VariableNames);

%% local training
mdl_loc = cellfun(@(f, x, y) f(x, y), fit_methods, x_train, y_train, 'UniformOutput', 0);
loc_acc = cellfun(@(m)mean(m.predict(x_test) == y_test), mdl_loc);

%% public label
y_public_all = cellfun(@(m)m.predict(x_public), mdl_loc, 'UniformOutput', 0);
y_public_all = [y_public_all{:}];
cats = unique(y_public_all);
y_public = cats((mean(y_public_all == cats(2), 2) > .5) + 1);

%% fed training
mdl_fed = cellfun(@(f, x, y)f([x; x_public], [y; y_public]), fit_methods, x_train, y_train, 'UniformOutput', 0);
fed_acc = cellfun(@(m)mean(m.predict(x_test) == y_test), mdl_fed);