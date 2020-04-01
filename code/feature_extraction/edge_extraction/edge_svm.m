% Script for implementing edge extraction for feature extraction 
% and SVM as classification 

clear all
close all 

%adding SVM paths
%addpath_recurse  ..\..\classification\svm\svm_km\
addpath ..\..\helper_functions\
addpath_recurse  ..\..\classification\

%loading 50% of training images
disp("Loading Training Images...");
disp("===========================");
num_imgs = 3000;
[images, labels] = load_image_database('..\..\..\', 'images.dataset');

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
    img = uint8(img);
    [edges, Ihor, Iver] = edgeextraction(img); 
    edge_matrix(i,:) = edges; 
end

% generate svm model from extracted edges and image labels 
disp("Creating SVM model from extracted edges...");
disp("==========================================");
edge_svm_model = svm_training(edge_matrix, training_label);

[test_images_row test_images_col] = size(test);
[edge_matrix_test] = zeros(test_images_row, test_images_col);
[rowSize_test colSize_test] = size(edge_matrix_test); 

%edge extraction on training images
disp("Extracting Edges on Test Images...");
disp("==================================");

for i = 1:rowSize_test
    
    img = reshape(test(i,:),160,96);
    img = uint8(img);
    [test_edges, test_Ihor, test_Iver] = edgeextraction(img); 
    edge_matrix_test(i,:) = test_edges;
    
end

[test_row test_col] = size(test);

% calculate predictions from the svm model on the test images 
for i = 1:test_row
    predictions(i,1) = svm_testing(edge_matrix_test(i,:),edge_svm_model);
end

%set the negative labels to 0 prior to comparison
test_label(test_label==-1)=0;

% compare and display the accuracy of the model 
disp("SVM Classification Accuracy...");
disp("============================");
comparison = (test_label == predictions);
accuracy = sum(comparison)/length(comparison);
disp(accuracy);

%saving edge extraction svm model
%save edge_svm_model edge_svm_model 
