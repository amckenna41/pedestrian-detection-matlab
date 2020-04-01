% Script for implementing edge extraction for feature extraction 
% and KNN as classification 

close all
clear all

%adding all required files and functions 
addpath ('..\..\helper_functions')
addpath_recurse  ..\..\classification\

%loading all images
disp("Loading Images...");
disp("=================");
num_imgs = 3000;
[images, labels] = load_image_database('..\..\..\', 'images.dataset');

%splitting dataset into training and testing
training = images(1:(num_imgs/2),:);
training_label = labels(1:(num_imgs/2));

test = images((num_imgs/2)+1:num_imgs,:);
test_label = labels((num_imgs/2)+1:num_imgs);

[imageRow imageCol] = size(training); 
[edge_matrix] = zeros(imageRow, imageCol); 
[rowSize colSize] = size(edge_matrix); 

%edge extraction on training images
disp("Extracting Edges on Training Images...");
disp("======================================");

for i = 1:rowSize
    
    img = reshape(training(i,:),160,96);
    [edges, Ihor, Iver] = edgeextraction(img); 
    edge_matrix(i,:) = edges; 
    
end

%create KNN model
disp("Creating KNN Model...");
disp("=====================");
edge_knn_model = NNtraining(edge_matrix, training_label); 

[test_images_row test_images_col] = size(test);
[edge_matrix_test] = zeros(test_images_row, test_images_col);
[rowSize_test colSize_test] = size(edge_matrix_test); 

%edge extraction on test images
disp("Extracting Edges on Test Images...");
disp("===================================");

for i = 1:rowSize_test
    
    img = reshape(test(i,:),160,96);
    [test_edges, test_Ihor, test_Iver] = edgeextraction(img); 
    edge_matrix_test(i,:) = test_edges; 
    
end

%getting size of matrix of test images
[test_row, test_col] = size(test); 

%negative image labels set to 0 
test_label(test_label==-1)=0;

disp("Starting KNN Classification...");
disp("==============================");

%matrix to store predictions 
predictions = zeros(test_row,1);
    
for j =1:size((test_row),1)
   predictions(j) = KNNTesting(edge_matrix_test(j,:),edge_knn_model);
end
    
%comparing results from svm testing function with test labels
disp("KNN Classification Accuracy...");
disp("==============================");
comparison = (test_label == predictions);
accuracy = sum(comparison)/length(comparison);
disp(accuracy);
    
%save model
%save edge_knn_model edge_knn_model 
