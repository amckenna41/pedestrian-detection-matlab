clear all;
close all;

% importing training images from current directory 
% fix relative pathing once file/directory strucuture is known
train_imgs = dir('*.jpg');
num_imgs = length(train_imgs);

% pull known dark and bright images from training images
dark_img = imread(train_imgs(1).name);
bright_img = imread(train_imgs(9).name);

% display 2 images and their histograms in figure
figure;
subplot(2,2,1);
imshow(dark_img);
subplot(2,2,2);
imshow(bright_img);
subplot(2,2,3);
histogram(dark_img, 'BinLimits', [0 256], 'BinWidth', 1);
subplot(2,2,4);
histogram(bright_img, 'BinLimits', [0 256], 'BinWidth', 1);

% apply brightness enhancement that will improve dark image, to both images
dark_img = enhanceBrightness(dark_img, 50);
bright_img = enhanceBrightness(bright_img, 50);

% re-plot figure to show changes
figure;
subplot(2,2,1);
imshow(dark_img);
subplot(2,2,2);
imshow(bright_img);
subplot(2,2,3);
histogram(dark_img, 'BinLimits', [0 256], 'BinWidth', 1);
subplot(2,2,4);
histogram(bright_img, 'BinLimits', [0 256], 'BinWidth', 1);
