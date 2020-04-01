% Script for implementing dimensionality reduction with lda for feature extraction 
% and SVM as classification (no pre-processing)

close all
clear all

%adding all required files and functions 
addpath ..\..\helper_functions
addpath_recurse  ..\..\classification\

%loading images
disp("Loading Image Dataset...");
disp("========================");
num_imgs = 3000;
[images, labels] = load_image_database('..\..\..\', 'images.dataset');

%splitting dataset into training and test
training = images(1:(num_imgs/2),:);
training_label = labels(1:(num_imgs/2));

test = images((num_imgs/2)+1:num_imgs,:);
test_label = labels((num_imgs/2)+1:num_imgs);

%dimensionality reduction on training images with X dimensions
disp("Dimensionality Reduction on training images...");
disp("==============================================");
[eigenVectors, eigenvalues, meanX, Xpca] = PrincipalComponentAnalysis(training,40);

disp("Linear Discriminat Analysis on reduced images...");
disp("================================================");
[eigenVectors_lda, eigenvalues_lda, meanX_lda, Xlda_train] = LDA(training_label,[],Xpca);

%creating svm model from lda images 
disp("Training SVM Model from LDA images...");
disp("=====================================");
svm_model = svm_training(Xlda_train, training_label);


%Test svm model on test images, convert to PCA then LDA space first 

for i=1:size(test,1)

    testNumber = test(i,:);
    %transform test image into PCA feature space
    pca_transform = (testNumber - meanX)*eigenVectors;
    %transform test image into LDA feature space
    lda_transform = (pca_transform - meanX_lda) * eigenVectors_lda;
    result = svm_testing(lda_transform, svm_model); 
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

%saving lda reduced dimensionality svm model 
save lda_svm svm_model
