clear; close all; clc;
%% Description:
%   This script extracts Legendre Moments from a set of image files using
%   user-defined moment functions. The script pads each image to square
%   shape, converts to grayscale if necessary, and computes the moment
%   features up to a specified order.
%
% Output:
%   For each moment function, the script creates a .mat file in the output
%   directory containing a matrix 'M', where each row is a feature vector
%   for an image, ending with its class label.
%
% Parameters to configure:
%   - LM_path: directory containing Legendre Moment function .m file.
%   - imagePath: directory containing input sub-images (.jpg).
%   - outputPath: directory where output .mat files will be saved.
%   - ord: vector of moment orders.
%   - classLabel: numeric class label to append to feature vectors.

%% Parameters
lmPath = 'LM_path'; % set LM.m path
imagePath = 'sub-image_path'; % set sub-images path
outputPath = 'output_path'; % set output path
ord = 10; % LM Order
classLabel = 1; % 1 for mouse, 0 for other

% Get moment function file
momentFiles = dir(fullfile(lmPath, '*.m'));
fn = momentFiles.name(1:end-2);
moment_func = str2func(fn);

% Get all image files
imageFiles = dir(fullfile(imagePath, '*.jpg'));
imageFiles([imageFiles.isdir]) = [];

LM = zeros(length(imageFiles), ord * ord + 1); % + 1 for class

for j = 1:length(imageFiles)
    fprintf('Processing function 1/%d, image %d/%d\n', length(momentFiles), j, length(imageFiles));

    % Read and preprocess image
    img = imread(fullfile(imagePath, imageFiles(j).name));
    if ndims(img) == 3
        img = rgb2gray(img);
    end
    img = double(img);

    % Pad to square
    max_dim = max(size(img));
    I_square = zeros(max_dim);
    I_square(1:size(img,1), 1:size(img,2)) = img;

    % Apply moment function
    m = moment_func(I_square, ord);
    LM(j,:) = [reshape(m, 1, []) classLabel];

end

% Save feature matrix
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end
save(fullfile(outputPath, [fn '-' num2str(classLabel) '.mat']), 'LM');