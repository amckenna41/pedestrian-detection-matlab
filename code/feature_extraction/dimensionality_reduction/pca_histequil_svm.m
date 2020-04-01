%%Script that creates an SVM model using histogram equilisation
%%as a pre-processing technique and pca dimensionality reduction for feaure extraction
%%and SVM as classification
clear all
close all 

%adding relevant function paths 
addpath ..\..\helper_functions\
addpath_recurse  ..\..\classification\
addpath_recurse ..\..\pre_processing

%loading dataset
disp("Loading Training Images...");
disp("==========================");
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

%hist equilisation on training images
disp("Histogram Equilisation on Training Images...");
disp("============================================");

for i = 1:rowSize
   
    enhanced_img = enhanceContrastHE(training(i,:));
    histequil_images(i,:) = enhanced_img; 
    
end

% dimensionality reduction on training images
disp("Dimensionality Reduction on Training Images...");
disp("==============================================");

[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(histequil_images,40);

%create dimensionality reduction SVM model on training images
disp("Creating SVM Model...");
disp("=====================");
svm_model = svm_training(Xpca, training_label);
% 
% %convert images into double 
% training = double(training); 
% test = double(test);

[test_images_row test_images_col] = size(test);
[histequil_test] = zeros(test_images_row, test_images_col);
[rowSize_test colSize_test] = size(histequil_test); 

%hist equlisation on test images
disp("Histogram Equilisation on Test Images...");
disp("========================================");

for i = 1:rowSize_test
   
    enhanced_img = enhanceContrastHE(test(i,:));
    histequil_test(i,:) = enhanced_img;
    
end

%create SVm model from test images and store classification results 
disp("Testing SVM model on test images...");
disp("===================================");

prediction = zeros(size(test,1),1);

for i=1:size(test,1)

    testNumber = test(i,:);
    pca = (testNumber - meanX)*eigenVectors;
    result = svm_testing(pca, svm_model);
    prediction(i,:,:) = result;

end

%negative image labels set to 0 
test_label(test_label==-1)=0;

%comparing results from svm testing function with test labels
disp(" SVM Classification Accuracy");
disp("============================");
comparison = (test_label == prediction);
accuracy = sum(comparison)/length(comparison);
disp(accuracy);

% ConfusionMatrix(test_label, prediction,1);

% %5-fold cross validation 
% disp(" 5-Fold Cross Validation Accuracy");
% disp("=================================");
% cv_svm_model = fitcsvm(Xpca, labels);
% model_cv_svm = crossval(svm_model, 'Kfold',5);
% results = kfoldPredict(model_cv_svm);
% comparison = (labels == results);
% accuracy = sum(comparison)/length(comparison);
% disp(accuracy);

%save model
save histequil_pca_svm svm_model 
