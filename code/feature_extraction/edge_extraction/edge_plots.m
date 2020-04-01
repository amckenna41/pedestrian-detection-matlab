%%script for plotting the edges images as shown in the report 

clear all
close all

%importing positive images from images/pos directory
addpath('..\..\helper_functions');
img_folder = dir('..\..\..\images\pos\*.jpg');
numFiles = length(img_folder);

image_size = 160*96;
edge_mat = zeros(numFiles,(image_size));

 
%randomly generate number that is used to display edges of image at index  
rand1 = randi(numFiles);

%convert image matrix into greyscale and read image at random index 
img = rgb2gray(imread(img_folder(rand1).name));

%carry out edge extraction on image at random index 
[edges, Ihor, Iver] = edgeextraction(img);
edges = reshape(edges, [160 96]);

%display total edges, horizontal edges and vertical edges 

figure;
subplot(221), imagesc(edges), title('All Edges')
subplot(222), imagesc(Ihor), title('Horizontal Edges')
subplot(223), imagesc(Iver), title('Vertical Edges')