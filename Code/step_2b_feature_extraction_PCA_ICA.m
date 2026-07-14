clear all, close all, clc
%% Description
% This script processes grayscale image data for feature extraction using
% Principal Component Analysis or Independent Component Analysis.
% It is structured in three steps:
%   1. Image Preprocessing:
%      - Reads all .jpg images from a specified class folder.
%      - Flattens each image into a 1D row vector and aggregates them into
%        a data matrix called 'DATA'.
%      - Saves the resulting matrix to disk.
%   2. Dataset Construction:
%      - Combines image data from classes 0 and 1 and add labels.
%      - Saves the labeled dataset as 'DATA.mat'.
%   3. Feature Extraction:
%      - Applies PCA to reduce dimensionality while retaining 95% variance.
%        Saves the PCA scores and explained variance.
%      - Applies FastICA to extract 100 independent components.
%        Saves the mixing matrix with class labels.

%% 1 - Image Preprocessing
% Run separately for classes 0 and 1
class = 1; % Class label: 1 = mouse trace, 0 = other
imagePath = 'sub-image_path'; % set sub-images path
outputPath = 'save vectorized image data'; % Path to save class .mat file

% Define full image directory
image_dir = fullfile(imagePath, num2str(class));
cd(image_dir);

% Read all image files
Files = dir('*.jpg');
Files([Files.isdir]) = [];

% Flattens each image into a 1D row vector and aggregates them into DATA
DATA = [];
for i = 1:length(Files)
    fprintf('Processing image %d of %d\n', i, length(Files));
    I = double(imread(Files(i).name));     % Convert image to double
    I_vec = reshape(I, 1, []);             % Flatten image to 1D vector
    
    % Dynamically pad rows for different image sizes
    [r1, c1] = size(DATA);
    [~, c2] = size(I_vec);
    M = zeros(r1 + 1, max(c1, c2));
    M(1:r1, 1:c1) = DATA;
    M(end, 1:c2) = I_vec;
    DATA = M;
end

% Save vectorized data for the class
cd(outputPath);
save([num2str(class) '.mat'], 'DATA', '-v7.3');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all, close all, clc
%% 2 - Merge Classes and Add Labels
% Load previously saved class data
dataPath = 'saved vectorized image data'; % Path to saved .mat files
cd(dataPath);
% Load data for both classes
if exist('0.mat', 'file') && exist('1.mat', 'file')
    load('0.mat');
    D = DATA; % Class 0
    load('1.mat'); % Class 1 in variable DATA
else
    error('Class data files not found. Please run step 1 for both classes.');
end

% Merge datasets
M = zeros(size(D,1) + size(DATA,1), max(size(D,2), size(DATA,2)));
M(1:size(D,1), 1:size(D,2)) = D;
M(1:size(D,1), end+1) = 0;  % Label for class 0

M(size(D,1)+1:end, 1:size(DATA,2)) = DATA;
M(size(D,1)+1:end, end) = 1;  % Label for class 1

save('DATA.mat', 'M', '-v7.3');

%% 3 - PCA Feature Extraction
fprintf('Running PCA...\n');
tic;
[coeff, score, latent, tsquared, explained, mu] = ...
    pca(M(:,1:end-1), 'NumComponents', 100);
time_pca = toc;

% Identify number of components explaining >95% variance
idx_95 = find(cumsum(explained) > 95, 1);

% Store PCA-reduced features with labels (IMPORTANT!!!)
% 1 - 95% variance 
PCA = [score(:,1:idx_95), M(:, end)];
% 2 - all scores
PCA = [score, M(:, end)];

% Save PCA results
save('PCA.mat', 'PCA', '-v7.3');

%% 4 - ICA Feature Extraction (FastICA)
% FastICA can be downloaded from
% https://research.ics.aalto.fi/ica/fastica/
fprintf('Running ICA...\n');
fasticaPath = 'FastICA path';  % Path where FastICA is installed
icaSavePath = 'Directory to save ICA'; % Path to save ICA results

cd(fasticaPath);
tic
[s, A, W] = fastica(M(:,1:end-1), ...
    'approach', 'symm', ...
    'g', 'gaus', ...
    'numOfIC', 100);
time_ica = toc;

% Store ICA features (mixing matrix) with labels
ICA = [A, M(:, end)];

% Save ICA results
cd(icaSavePath);
save('ICA.mat', 'ICA', '-v7.3');

fprintf('Feature extraction complete. PCA time: %.2fs | ICA time: %.2fs\n', ...
    time_pca, time_ica);