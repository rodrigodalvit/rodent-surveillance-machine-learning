clear all, close all, clc
%% Description
% This MATLAB script performs classification using the k-Nearest Neighbors
% (k-NN) algorithm on datasets derived from ICA, PCA, and LM.
% It evaluates performance by iteratively training and testing k-NN models
% using different numbers of features and computes performance metrics such
% as accuracy, sensitivity, specificity, loss, and MSE.
% This code generates Figure 4

%% k-NN with ICA, PCA, LM (Different Numbers of Features)
cd('path to LM.mat, PCA.mat, and ICA.mat');
Files = dir('*.mat');
Files([Files.isdir]) = [];

% Initialize containers
vec_size = 80; % training size (%)
DATA_new = [];

for i = 1:length(Files)
    fn = Files(i).name;
    loaded = load(fn);    
    % Try to access different structures if available
    if isfield(loaded, 'ICA')
        DATA = loaded.ICA;
    elseif isfield(loaded, 'PCA')
        DATA = loaded.PCA;
    elseif isfield(loaded, 'LM')
        DATA = loaded.LM;
    else
        continue;
    end
    
    % Remove zero-columns (except label column)
    zeroCols = find(all(DATA(:,1:end-1) == 0));
    DATA(:, zeroCols) = [];

    for len = 1:100 % set number of features
        data = [DATA(:,1:len) DATA(:,end)];
        for j = 1:50 % training 50 times
            [~, c] = size(data);
            Aparts = cell(3,1);
            Bparts = cell(3,1);
            % Balanced split per class
            for k = min(data(:,end)):max(data(:,end))
                idx = find(data(:,end)==k);
                idx = idx(randperm(length(idx)));  % Shuffle
                numTrain = round((vec_size/100)*length(idx));
                Aparts{k+1} = data(idx(1:numTrain), :);
                Bparts{k+1} = data(idx(numTrain+1:end), :);
            end
            train = cell2mat(Aparts);
            test  = cell2mat(Bparts);

            %% Train k-NN Classifier
            knn = fitcknn(train(:,1:end-1), train(:,end), ...
                          'NumNeighbors', 5, ...
                          'Distance', 'euclidean');

            [class_knn, ~] = predict(knn, test(:,1:end-1));
            Class_knn = class_knn - test(:,end);

            %% Metrics
            confmat = confusionmat(test(:,end), class_knn);
            TP(:,j) = confmat(2,2);
            TN(:,j) = confmat(1,1);
            FP(:,j) = confmat(1,2);
            FN(:,j) = confmat(2,1);

            Acc_knn(:,j) = mean(Class_knn == 0) * 100;
            Sens(:,j) = TP(:,j) / (TP(:,j) + FN(:,j));
            Spec(:,j) = TN(:,j) / (TN(:,j) + FP(:,j));
            Class_Loss(:,j) = loss(knn, train(:,1:end-1), train(:,end), 'LossFun', 'crossentropy');
            err(:,j) = immse(test(:,end), class_knn);
        end
        % Store mean & std of metrics
        mean_acc(i,len)  = mean(Acc_knn);
        mean_sen(i,len)  = mean(Sens);
        mean_spe(i,len)  = mean(Spec);
        mean_loss(i,len) = mean(Class_Loss);
        mean_mse(i,len)  = mean(err);

        std_acc(i,len)  = std(Acc_knn);
        std_sen(i,len)  = std(Sens);
        std_spe(i,len)  = std(Spec);
        std_loss(i,len) = std(Class_Loss);
        std_mse(i,len)  = std(err);
        
        clc; % Refresh display
    end
end

%% Plot Accuracy
ln = 1:size(mean_acc, 2);
colors = [0.47 0.67 0.19; 0.85 0.33 0.10; 0.6 0.2 0.8];

f = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
hold on;
plotHandles = gobjects(length(Files), 1);

for i = 1:length(Files)
    plotHandles(i) = plot(ln, mean_acc(i,:), '-', 'Color', colors(i,:), 'LineWidth', 2);
    fill([ln, fliplr(ln)], ...
         [mean_acc(i,:) + std_acc(i,:), fliplr(mean_acc(i,:) - std_acc(i,:))], ...
         colors(i,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
end

annotation('textbox', [0.05, 0.93, 0.05, 0.05], ...
           'String', '(a)', 'FontSize', 26, ...
           'FontWeight', 'bold', 'LineStyle', 'none');

legend(plotHandles, 'ICA', 'LM', 'PCA', 'Location', 'southeast', ...
       'FontSize', 20, 'Box', 'off');

xlabel('Number of Components', 'FontSize', 26);
ylabel('Accuracy (%)', 'FontSize', 26);
set(gca, 'FontSize', 16, 'LineWidth', 1.5);
xticks(0:10:100);
xlim([min(ln) max(ln)]);
grid on; box off;

print(f, 'Figure_4_accuracy', '-dpng', '-r300');
close(f); clc;

%% Plot Other Metrics
f = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);

metrics = {'Specificity', 'Sensitivity', 'Loss', 'MSE'};
data_mean = {mean_spe, mean_sen, mean_loss, mean_mse};
data_std  = {std_spe, std_sen, std_loss, std_mse};

for subplotIdx = 1:4
    subplot(2, 2, subplotIdx); hold on;
    
    for i = 1:length(Files)
        plotHandles(i) = plot(ln, data_mean{subplotIdx}(i,:), '-', ...
                              'Color', colors(i,:), 'LineWidth', 2);
        fill([ln, fliplr(ln)], ...
             [data_mean{subplotIdx}(i,:) + data_std{subplotIdx}(i,:), ...
              fliplr(data_mean{subplotIdx}(i,:) - data_std{subplotIdx}(i,:))], ...
             colors(i,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    end

    xlabel('Number of Components', 'FontSize', 26);
    ylabel(metrics{subplotIdx}, 'FontSize', 26);
    set(gca, 'FontSize', 16, 'LineWidth', 1.2);
    xticks(0:20:100);
    xlim([min(ln) max(ln)]);
    grid on; box off;
end

annotation('textbox', [0.05, 0.93, 0.05, 0.05], ...
           'String', '(b)', 'FontSize', 26, ...
           'FontWeight', 'bold', 'LineStyle', 'none');

lgd = legend(plotHandles, 'ICA', 'LM', 'PCA', ...
             'Orientation', 'horizontal', ...
             'FontSize', 18, 'Box', 'on');
lgd.Position = [0.45, 0.02, 0.15, 0.05];

print(f, 'Figure_4_statistics', '-dpng', '-r300');
close(f); clc;