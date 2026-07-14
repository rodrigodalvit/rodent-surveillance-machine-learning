clear all, close all, clc
%% Description
% This MATLAB script performs classification using the k-Nearest Neighbors
% (k-NN) algorithm on datasets derived from ICA, PCA, and LM.
% It evaluates performance by iteratively training and testing k-NN models
% using different training sizes (105 - 80%) and computes performance
% metrics such as accuracy, sensitivity, specificity, loss, and MSE.
% It plots ROC curves (at 80% training)
% This code generates Figure 5 and 6

%% k-NN with ICA, PCA, LM (Varying Training Size - 10% to 80%)
cd('path to LM.mat, PCA.mat, and ICA.mat');
Files = dir('*.mat'); 
Files([Files.isdir]) = [];  % Remove folders

% Initialize
len = 22;                         % Number of features to retain
vec_range = 10:80;               % Percentage of training data
clr = [0.47, 0.67, 0.19;         % Color palette for plots
       0.85, 0.33, 0.10;
       0.6, 0.2, 0.8];

for i = 1:length(Files)
    % Load and extract valid data structure
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

    % Remove all-zero columns (except class label)
    DATA(:, all(DATA(:,1:end-1)==0)) = [];

    count = 1;
    for vec_size = vec_range
        % Data preparation: take first N features + class label
        data = [DATA(:, 1:len), DATA(:, end)];

        for j = 1:50  % Repeat 50 times for averaging
            [r, c] = size(data);
            Aparts = cell(3,1); Bparts = cell(3,1);

            for k = min(data(:,end)):max(data(:,end))
                idx = find(data(:,end) == k);
                idx = idx(randperm(length(idx)));
                numTrain = round((vec_size / 100) * length(idx));
                Aparts{k+1} = data(idx(1:numTrain), :);
                Bparts{k+1} = data(idx(numTrain+1:end), :);
            end

            train = cell2mat(Aparts);
            test = cell2mat(Bparts);

            %% Train k-NN model
            knn = fitcknn(train(:,1:end-1), train(:,end), ...
                'NumNeighbors', 5, 'Distance', 'euclidean');
            [class_knn, score_knn] = predict(knn, test(:,1:end-1));

            %% Metrics
            confmat = confusionmat(test(:,end), class_knn);
            TP(:,j) = confmat(2,2); TN(:,j) = confmat(1,1);
            FP(:,j) = confmat(1,2); FN(:,j) = confmat(2,1);

            Acc_knn(:,j) = mean(class_knn == test(:,end)) * 100;
            Sens(:,j) = TP(:,j) / (TP(:,j) + FN(:,j));
            Spec(:,j) = TN(:,j) / (TN(:,j) + FP(:,j));
            Class_Loss(:,j) = loss(knn, train(:,1:end-1), train(:,end), ...
                                   'LossFun', 'crossentropy');
            err(:,j) = immse(test(:,end), class_knn);

            %% ROC and AUC plotting for best training size (80%)
            if vec_size == 80 && j == 1
                f = figure('Units','normalized','OuterPosition',[0 0 1 1]);
                title([fn(1:end-4)], 'FontSize', 38);

                rocObj = rocmetrics(test(:,end), score_knn, knn.ClassNames);
                h1 = plot(rocObj, 'LineWidth', 8, 'ClassNames', knn.ClassNames(1)); hold on;
                h2 = plot(rocObj, 'LineWidth', 8, 'ClassNames', knn.ClassNames(2));

                legend([h1, h2], {'NI', 'PI'}, 'Location', 'northeastoutside', 'FontSize', 34);
                xlabel('False Positive Rate'); ylabel('True Positive Rate');

                % Add metric summary to plot
                metricsText = sprintf(['NI AUC: %.4f\nPI AUC: %.2f\nAccuracy: %.2f\n' ...
                    'Specificity: %.2f\nSensitivity: %.2f\nLoss: %.2f\nMSE: %.2f'], ...
                    rocObj.AUC(1), rocObj.AUC(2), Acc_knn(:, j), ...
                    Spec(:, j), Sens(:, j), Class_Loss(:, j), err(:, j));
                text(0.99, 0.02, metricsText, 'Units','normalized', ...
                     'FontSize', 34, 'BackgroundColor', 'white', ...
                     'EdgeColor', 'black', 'Margin', 5, ...
                     'VerticalAlignment','bottom', 'HorizontalAlignment','right');
                ax = gca; ax.FontSize = 34;
                xticks(0:0.25:1); yticks(0:0.25:1);
                grid on; box off;

                print(f, ['Figure_6_' fn(1:end-4) '_auc'], '-dpng', '-r300');
                close(f);
            end
        end
        % Average across 3 repetitions
        mean_acc(i,count)  = mean(Acc_knn);
        mean_sen(i,count)  = mean(Sens);
        mean_spe(i,count)  = mean(Spec);
        mean_loss(i,count) = mean(Class_Loss);
        mean_mse(i,count)  = mean(err);

        std_acc(i,count)  = std(Acc_knn);
        std_sen(i,count)  = std(Sens);
        std_spe(i,count)  = std(Spec);
        std_loss(i,count) = std(Class_Loss);
        std_mse(i,count)  = std(err);

        count = count + 1;
        clc;
    end
end

%% Plot Mean Accuracy ± STD
ln = vec_range;
f = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
hold on;
for i = 1:length(Files)
    plot(ln, mean_acc(i,:), '-', 'Color', clr(i,:), 'LineWidth', 2);
    fill([ln fliplr(ln)], ...
         [mean_acc(i,:)+std_acc(i,:) fliplr(mean_acc(i,:)-std_acc(i,:))], ...
         clr(i,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
end
annotation('textbox', [0.05, 0.93, 0.05, 0.05], ...
    'String', '(a)', 'FontSize', 26, 'FontWeight', 'bold', ...
    'LineStyle', 'none', 'HorizontalAlignment', 'left');
xlabel('Training Size (%)', 'FontSize', 26);
ylabel('Accuracy (%)', 'FontSize', 26);
set(gca, 'FontSize', 16, 'LineWidth', 1.5); xticks(0:10:100); xlim([min(ln) max(ln)]);
grid on; box off;
print(f, 'Figure_5_accuracy', '-dpng', '-r300');
close(f);

%% Plot Other Stats (Specificity, Sensitivity, Loss, MSE)
f = figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);
metrics = {'Specificity', 'Sensitivity', 'Loss', 'MSE'};
data_mean = {mean_spe, mean_sen, mean_loss, mean_mse};
data_std = {std_spe, std_sen, std_loss, std_mse};

for subplotIdx = 1:4
    subplot(2, 2, subplotIdx); hold on;
    for i = 1:length(Files)
        plot(ln, data_mean{subplotIdx}(i,:), '-', ...
            'Color', clr(i,:), 'LineWidth', 2);
        fill([ln fliplr(ln)], ...
             [data_mean{subplotIdx}(i,:)+data_std{subplotIdx}(i,:) ...
              fliplr(data_mean{subplotIdx}(i,:)-data_std{subplotIdx}(i,:))], ...
             clr(i,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none');
    end
    xlabel('Training Size (%)'); ylabel(metrics{subplotIdx});
    set(gca, 'FontSize', 14); grid on; box off;
end

print(f, 'Figure_5_statistics', '-dpng', '-r300');
close(f);
