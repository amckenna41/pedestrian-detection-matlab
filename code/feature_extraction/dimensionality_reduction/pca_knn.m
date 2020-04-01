% Script for implementing dimensionality reduction (pca) for feature extraction 
% and KNN as classification 

close all
clear all

%adding all required files and functions 
addpath ('..\..\helper_functions')
addpath_recurse  ..\..\classification\

%array of k values to be used for KNN
k = [3, 5, 8, 10, 12, 15]; 
[row col] = size(k);

%loading images
disp("Loading  Images...");
disp("==================");
num_imgs = 3000;
[images, labels] = load_image_database('..\..\..\', 'images.dataset');

%splitting dataset into training and testing
training = images(1:(num_imgs/2),:);
training_label = labels(1:(num_imgs/2));

test = images((num_imgs/2)+1:num_imgs,:);
test_label = labels((num_imgs/2)+1:num_imgs);

[imageRow imageCol] = size(training); 
[histequil_images] = zeros(imageRow, imageCol); 
[rowSize colSize] = size(histequil_images); 


%dimensionality reduction on training images with X dimensions
disp("Dimensionality reduction on training images...");
disp("==============================================");
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(training,40);

%create KNN model from training images and labels
reduced_dimen_knn_model = NNtraining(Xpca, training_label); 

predictions = zeros(num_imgs/2,1);

%KNN testing on test images
disp("Starting KNN Testing on test images...");
disp("======================================");
for i =1:(num_imgs/2)
    testNumber = test(i,:);
    pca = (testNumber - meanX)*eigenVectors;
    result= KNNTesting(pca, reduced_dimen_knn_model);
    predictions(i) = result;
end

%negative image labels set to 0 
test_label(test_label==-1)=0;
 
%comparing results from svm testing function with test labels
disp("KNN Classification Accuracy...");
disp("==============================");
comparison = (test_label == predictions);
accuracy = sum(comparison)/length(comparison);
disp(accuracy);
    
% save model
% save dimensionality_reduction_pca_knn reduced_dimen_knn_model 
