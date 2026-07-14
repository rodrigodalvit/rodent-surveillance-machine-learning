clear all; close all; clc

%% Code Description: Image Preprocessing
% This MATLAB script is used to preprocess and divide images for downstream
% analysis and classification.
%
% It performs the following main steps:
%   - Applies a global threshold to grayscale images to reduce intraclass variance,
%   - Divides each image into 25 equal-sized sub-images (5 vertical × 5 horizontal),
%   - Saves each sub-image individually for further processing,
%
% (Tip) To facilitate image classification, it is recommended to
% divide the input images into directories according to category:
%   - 1 for mouse trace
%   - 0 for other animals

%% User Inputs (set these paths and threshold manually)
inputDir = 'define directory path with jpg files';      % Directory containing input .jpg images
outputDir = 'define directory path to save sub-images'; % Directory where sub-images will be saved
thresholdValue = 129.853;  % Define global threshold value (based on the first image)

showImages = false;  % <<< Set to true if you want to visualize, false if not

%% Create batched sub-images
cd(inputDir)
Files = dir('*.jpg');
Files([Files.isdir]) = [];

% Loop through each image
for i = 1:length(Files)
    cd(inputDir)
    
    %% Read and preprocess image
    fn = Files(i).name;
    I = imread(fn);
    I = rgb2gray(I);   
    
    % Apply global threshold by normalizing intensity
    I = I * (thresholdValue / mean2(I));  % Normalize image using mean intensity

    %% Divide image into 5x5
    [rows, columns] = size(I);
    vet = 5; hor = 5;
    topRows = round(linspace(1, rows+1, vet + 1));
    leftColumns = round(linspace(1, columns+1, hor + 1));
    
    % Display only if flag is set
    if showImages
        figure, imshow(I), title(['Grid Overlay: ' fn])
        hold on
        for k = 1:length(topRows)
            yline(topRows(k), 'Color', 'y', 'LineWidth', 2);
        end
        for k = 1:length(leftColumns)
            xline(leftColumns(k), 'Color', 'y', 'LineWidth', 2);
        end
        hold off
    end

    %% Extract and save sub-images     
    plotCounter = 1;
    count = 1;
    if showImages
        figure('Name', ['Sub-images of: ' fn]);
    end
    
    for row = 1:length(topRows)-1
        row1 = topRows(row);
        row2 = topRows(row + 1) - 1;
        for col = 1:length(leftColumns)-1
            col1 = leftColumns(col);
            col2 = leftColumns(col + 1) - 1;
            
            subImage = I(row1:row2, col1:col2);
            
            if showImages
                subplot(vet, hor, plotCounter);
                imshow(subImage);
                title(sprintf('Rows %d-%d, Cols %d-%d', row1, row2, col1, col2));
                plotCounter = plotCounter + 1;
            end
            
            % Save sub-image
            cd(outputDir)
            baseName = fn(1:end-4);
            imwrite(subImage, fullfile(outputDir, [baseName '-(' num2str(count) ').jpg']));
            count = count + 1;
        end
    end
    
    if ~showImages
        close all
    end
end