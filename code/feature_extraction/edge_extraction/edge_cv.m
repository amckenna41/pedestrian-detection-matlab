% Script for implementing edge extraction for feature extraction 
% with cross-validation

clear all
close all 

%adding SVM paths
%addpath_recurse  ..\..\classification\svm\svm_km\
addpath ..\..\helper_functions\
addpath_recurse  ..\..\classification\
addpath_recurse  ..\..\pre_processing\

%loading images
disp("Loading Images...");
disp("=================");
num_imgs = 3000;
[images, labels] = load_image_database('..\..\..\', 'images.dataset');

[imageRow imageCol] = size(images);

[edge_matrix] = zeros(imageRow, imageCol); 
[rowSize colSize] = size(edge_matrix); 

%edge extraction on training images
disp("Extracting Edges on Images...");
disp("=============================");

for i = 1:rowSize
    
    img = reshape(images(i,:),160,96);
    img = uint8(img);
    [edges, Ihor, Iver] = edgeextraction(img); 
    edge_matrix(i,:) = edges; 
end

%set the negative labels to 0 prior to comparison
labels(labels==-1)=0;

%5-fold cross validation 
disp("5-Fold Cross Validation Accuracy");
disp("================================");
cv_svm_model = fitcsvm(edge_matrix, labels);
model_cv_edge = crossval(cv_svm_model, 'Kfold',5);
results = kfoldPredict(model_cv_edge);
comparison = (labels == results);
accuracy = sum(comparison)/length(comparison);
disp(accuracy);

%saving edge extraction CV model
%save edge_crossval model_cv_edge 
