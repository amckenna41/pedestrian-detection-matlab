%script that uses dimensionality reduction (pca) using cross-validation
%along with various pre-processing techniques 

clear all
close all

%adding all required files and functions 
addpath ..\..\helper_functions
addpath_recurse  ..\..\classification\
addpath_recurse  ..\..\pre_processing\

%loading all images
disp("Loading Image Dataset...");
disp("========================");
[images, labels] = load_image_database('..\..\..\', 'images.dataset');

%pre-processing for all images
disp("Implementing Pre-processing on images...");
disp("=====================================");

%Uncomment relevant code to add pre-processing technique%

% %brightness extraction on  images
% disp("Brightness Enhancement on Images...");
% disp("===================================");
% 
% [imageRow imageCol] = size(images); 
% [brightness_images] = zeros(imageRow, imageCol); 
% [rowSize colSize] = size(brightness_images); 
% c = 20; 
% 
% for i = 1:rowSize
%     
%     img = reshape(images(i,:),160,96);
%     img = uint8(img);
%     enhanced_img = enhanceBrightness(img, c);
%     reshaped_img = reshape(enhanced_img, [1 colSize]);
%     brightness_images(i,:) = reshaped_img; 
%     
% end
% 
% %hist equilisation on images
% disp("Histogram Equilisation on Images...");
% disp("===================================");
% 
% [imageRow imageCol] = size(images); 
% [histequil_images] = zeros(imageRow, imageCol); 
% [rowSize colSize] = size(histequil_images); 
% 
% for i = 1:rowSize
%    
%     enhanced_img = enhanceContrastHE(images(i,:));
%     histequil_images(i,:) = enhanced_img; 
%     
% end


% %Power Law on images
% disp("Power Law on Images...");
% disp("======================");
% 
% [imageRow imageCol] = size(images); 
% [power_images] = zeros(imageRow, imageCol); 
% [rowSize colSize] = size(power_images); 
% 
% gamma = 0.5;
% % gamma = 2; 
% 
% for i = 1:rowSize
%     
%     enhanced_img = enhanceContrastPL(uint8(images(i,:)), gamma); 
%     power_images(i,:) = enhanced_img; 
%     
% end

%dimensionality reduction (PCA) all images
%uncomment relevant code to add pre-processing technique
disp("Dimensionality Reduction on images...");
disp("=====================================");

[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(images,40);
%[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(brightness_images,10);
%[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(histequil_images,10);
%[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(power_images,10);

%set the negative labels to 0 prior to comparison
labels(labels==-1)=0;

%5-fold cross validation 
disp("5-Fold Cross Validation Accuracy");
disp("================================");
cv_svm_model = fitcsvm(Xpca, labels);
model_cv_svm = crossval(cv_svm_model, 'Kfold',5);
results = kfoldPredict(model_cv_svm);
comparison = (labels == results);
accuracy = sum(comparison)/length(comparison);
disp(accuracy);

%saving model
save pca_crossval model_cv_svm