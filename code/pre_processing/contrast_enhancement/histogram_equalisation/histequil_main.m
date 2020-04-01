%Main script for calling histogram equilisation function on all images

clear all
close all

%importing positive images from images/pos directory
addpath('..\..\..\helper_functions');
img_folder = dir('..\..\..\..\images\pos\*.jpg');

numFiles = length(img_folder);
image_size = 160*96;
histeq_mat = zeros(numFiles,(image_size));

% for each image file, read image, call histogram equilisation function,
% reshape image so that each row has image pixels from one image,
% add reshaped array into row in HE matrix

for k=1:numFiles

    file_name = img_folder(k).name;
    image = imread(file_name);
    image = rgb2gray(image);
    enhanced_image = enhanceContrastHE(image);
    reshaped_image = reshape(enhanced_image, [1,image_size]);
    histeq_mat(k,:) = reshaped_image;

end


%Example images before and after histogram equilisation applied
%change image index to select different enhanced image 

figure;
subplot(2,2,1)
imshow(rgb2gray(imread(img_folder(2).name)));
title('Normal Image');
subplot(2,2,2)
imshow(uint8(reshape(histeq_mat(2,:), [160, 96])))
title('Enhanced Image');
subplot(2,2,3)
imshow(rgb2gray(imread(img_folder(20).name)));
title('Normal Image');
subplot(2,2,4)
imshow(uint8(reshape(histeq_mat(20,:), [160, 96])));
title('Enhanced Image');

sgtitle('Histogram Equilisation Pre-Processing','Color','blue');
sgt.FontSize = 20; 

%Further histogram equilisation examples, 
%change image index to change image being enhanced

% figure;
% subplot(2,2,1)
% imshow(rgb2gray(imread(img_folder(15).name)));
% title('Normal Image');
% subplot(2,2,2)
% imshow(uint8(reshape(histeq_mat(15,:), [160, 96])))
% title('Enhanced Image');
% subplot(2,2,3)
% imshow(rgb2gray(imread(img_folder(48).name)));
% title('Normal Image');
% subplot(2,2,4)
% imshow(uint8(reshape(histeq_mat(48,:), [160, 96])));
% title('Enhanced Image');
% sgtitle('Histogram Equilisation Pre-Processing','Color','blue');
% sgt.FontSize = 20; 
