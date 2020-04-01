% Script for implementing dimensionality reduction(pca) for feature extraction 
% and SVM as classification with no pre-processing

close all
clear all

%adding all required files and functions 
addpath ..\..\helper_functions
addpath_recurse  ..\..\classification\

%loading images
disp("Loading Training Images...");
disp("===========================");
num_imgs = 3000;
[images, labels] = load_image_database('..\..\..\', 'images.dataset');

training = images(1:(num_imgs/2),:);
training_label = labels(1:(num_imgs/2));

test = images((num_imgs/2)+1:num_imgs,:);
test_label = labels((num_imgs/2)+1:num_imgs);

%dimensionality reduction on training images with X dimensions
disp("Dimensionality Reduction on training images...");
disp("==============================================");
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(training,40);

%create SVM model
svm_model = svm_training(Xpca, training_label);

% %convert images into double 
% training = double(training); 
% test = double(test);

%create SVM model from test images and store classification results 
disp("Test SVM Model on test images...");
disp("================================");

prediction = zeros(size(test,1),1);

for i=1:size(test,1)

    testNumber = test(i,:);
    pca = (testNumber - meanX)*eigenVectors;
    result = svm_testing(pca, svm_model);
    prediction(i,:,:) = result;
    %prediction(i,1) = svm_testing(pca, svm_model);

end

%negative image labels set to 0 
test_label(test_label==-1)=0;

%comparing results from svm testing function with test labels
disp(" SVM Classification Accuracy");
disp("============================");
comparison = (test_label == prediction);
accuracy = sum(comparison)/length(comparison);
disp(accuracy);

%saving reduced dimensionality svm model 
save pca_svm_model svm_model