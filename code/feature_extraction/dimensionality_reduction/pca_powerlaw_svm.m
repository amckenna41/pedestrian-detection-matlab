%%Script that creates an SVM model using power law
%%as a pre-processing technique and dimensionality reduction (pca) for feaure extraction
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
[power_images] = zeros(imageRow, imageCol); 
[rowSize colSize] = size(power_images); 

%gamma = 0.5;
gamma = 2; 

%Power Law on training images
disp("Power Law on Training Images...");
disp("===============================");

for i = 1:rowSize
    
    enhanced_img = enhanceContrastPL(uint8(training(i,:)), gamma); 
    power_images(i,:) = enhanced_img; 
    
end

% dimensionality reduction on training images
disp("Dimensionality Reduction on Training Images...");
disp("==============================================");

[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(power_images,40);


%create SVM extraction model from enhanced training mages
disp("Creating SVM Model...");
disp("=====================");
svm_model = svm_training(Xpca, training_label);


[test_images_row test_images_col] = size(test);
[power_images_test] = zeros(test_images_row, test_images_col);
[rowSize_test colSize_test] = size(power_images_test); 

% %convert images into double 
% training = double(training); 
% test = double(test);

%power law for test images
disp("Power Law on Test Images...");
disp("===========================");

for i = 1:rowSize_test
    
    enhanced_img = enhanceContrastPL(uint8(test(i,:)), gamma); 
    power_images_test(i,:) = enhanced_img; 
    
end


%test SVM model on test images and store classification results 
disp("Testing SVM Model on test images...");
disp("===================================");

prediction = zeros(size(test,1),1);

for i=1:size(test,1)

    testNumber = test(i,:);
    pca = (testNumber - meanX)*eigenVectors;
    result = svm_testing(pca, svm_model);
    prediction(i,:,:) = result;

end


%set the negative labels to 0 prior to comparison
test_label(test_label==-1)=0;

% compare and display the accuracy of the model 
disp("SVM Classification Accuracy...");
disp("==============================");
comparison = (test_label == prediction);
accuracy = sum(comparison)/length(comparison);
disp(accuracy);

% %5-fold cross validation 
% disp("5-Fold Cross Validation Accuracy");
% disp("================================");
% cv_svm_model = fitcsvm(Xpca, test_label);
% model_cv_svm = crossval(cv_svm_model, 'Kfold',5);
% results = kfoldPredict(model_cv_svm);
% comparison = (test_label == results);
% accuracy = sum(comparison)/length(comparison);
% disp(accuracy);

%saving model 
save powerlaw_pca_svm svm_model
