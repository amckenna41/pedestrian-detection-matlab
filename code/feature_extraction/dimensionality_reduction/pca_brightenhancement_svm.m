%%Script that creates an SVM model using brightness enhancement
%%as a pre-processing technique and dimensionality reduction (pca) for feaure extraction
%%and SVM as classification
clear all
close all 

%adding relevant function paths 
addpath ..\..\helper_functions\
addpath_recurse  ..\..\classification\
addpath_recurse ..\..\pre_processing

%loading image dataset
disp("Loading Image Dataset...");
disp("========================");
num_imgs = 3000;
[images, labels] = load_image_database('..\..\..\', 'images.dataset');

%splitting dataset into training and test
training = images(1:(num_imgs/2),:);
training_label = labels(1:(num_imgs/2));

test = images((num_imgs/2)+1:num_imgs,:);
test_label = labels((num_imgs/2)+1:num_imgs);

[imageRow imageCol] = size(training); 
[brightness_images] = zeros(imageRow, imageCol); 
[rowSize colSize] = size(brightness_images); 
c = [10, 15, 20, 40, 50];
%c=20 proved to give the greatest increase in accuracy 

%brightness extraction on training images
disp("Brightness Enhancement on Training Images...");
disp("============================================");

for i = 1:rowSize
    
    img = reshape(training(i,:),160,96);
    img = uint8(img);
    enhanced_img = enhanceBrightness(img, c(3));
    reshaped_img = reshape(enhanced_img, [1 colSize]);
    brightness_images(i,:) = reshaped_img; 
    
end

% dimensionality reduction on training images
disp("Dimensionality Reduction on Training Images...");
disp("==============================================");

[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(brightness_images,40);

%create SVM model
disp("Creating SVM Model...");
disp("=====================");
svm_model = svm_training(Xpca, training_label);

% %convert images into double 
% training = double(training); 
% test = double(test);

[imageRow imageCol] = size(test); 
[brightness_test_images] = zeros(imageRow, imageCol); 
[rowSize colSize] = size(brightness_test_images); 

%brightness extraction on test images
disp("Brightness Enhancement on Test Images...");
disp("========================================");

for i = 1:rowSize
    
    img = reshape(test(i,:),160,96);
    img = uint8(img);
    enhanced_img = enhanceBrightness(img, c(3));
    reshaped_img = reshape(enhanced_img, [1 colSize]);
    brightness_test_images(i,:) = reshaped_img; 
    
end
% 
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

% %5-fold cross validation 
% disp(" 5-Fold Cross Validation Accuracy");
% disp("=================================");
% cv_svm_model = fitcsvm(Xpca, test_label);
% model_cv_svm = crossval(cv_svm_model, 'Kfold',5);
% results = kfoldPredict(model_cv_svm);
% comparison = (test_label == results);
% accuracy = sum(comparison)/length(comparison);
% disp(accuracy);

save brightenhancement_pca_svm svm_model 
