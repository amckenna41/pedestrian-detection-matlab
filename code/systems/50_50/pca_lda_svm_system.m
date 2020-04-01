clear all;
close all;

% this script is a full machine learning pipeline which does pre
% processing, feature extraction, training and testing, and outputs an 
% accuracy %age for the system. The training/testing method used is 50:50. 

% the idea for this script is that the only lines of code which need
% altered to change the pipeline components are where the function handles
% for each pipeline stage are set (lines 25-28)

% Options for pipeline components:
% Pre-processing: [enhanceContrastHE, false]
% Feature extraction: [full_image, hog_feature_vector, edgeextraction, PrincipalComponentAnalysis]
% Training: [svm_training, NNtraining]
% Testing: [svm_testing, NNTesting, KNNTesting]

% assign variables for system
num_imgs = 3000;
dataset_file = 'images.dataset';
dataset_dir = '..\..\..\';

% set function handles for each stage of the system
pre_process = false;
feature_extraction = @PrincipalComponentAnalysis;
training = @svm_training;
testing = @svm_testing;

% add paths needed for system
addpath('..\..\helper_functions\');
addpath_recurse('..\..\pre_processing\');
addpath_recurse('..\..\classification\');
addpath_recurse('..\..\feature_extraction\');

% load full dataset
disp("Loading images...");
[images, labels] = load_image_database(dataset_dir, dataset_file);

%%%%%%%%%% PRE-PROCESSING %%%%%%%%%%
% apply data pre-processing technique and store output images
if ~isequal(pre_process, false)
    disp("Pre-processing...");
    for i = 1:num_imgs
        images(i,:) = pre_process(images(i,:));
    end
end

% split data into 50% training and 50% testing
images_train = images(1:(num_imgs/2),:);
labels_train = labels(1:(num_imgs/2));
images_test = images(((num_imgs/2))+1:num_imgs,:);
labels_test = labels(((num_imgs/2))+1:num_imgs);

%%%%%%%%%% FEATURE EXTRACTION %%%%%%%%%%
disp("Extracting features...");
%optimal principal components number is 40 
pc = 40; 

%extracting features into PCA feature space
[eigenVectors_train, eigenValues_train, meanX_train, Xpca_train] = feature_extraction(images_train, pc);
%extracting features into LDA feature space
[eigenVectors_lda, eigenValues_lda, meanX_lda, Xpca_lda] = LDA(labels_train,[],Xpca_train);


%%%%%%%%%% TRAIN & TEST MODEL %%%%%%%%%%
disp("Training model...");
model = training(Xpca_lda, labels_train);
predictions = zeros(num_imgs/2, 1);
disp("Testing model...");
for i = 1:(num_imgs/2)
    disp(i);
    testNumber = images_test(i,:);
    %extracting test images into pca feature space
    pca = (testNumber - meanX_train)*eigenVectors_train;
    %extracting test images into lda feature space
    lda = (pca - meanX_lda)*eigenVectors_lda; 
    result = testing(lda, model);
    predictions(i) = result; 
end

%%%%%%%%%% TEST MODEL ACCURACY %%%%%%%%%%
% calculate accuracy and display
comparison = (labels_test==predictions);
accuracy = sum(comparison)/length(comparison);

%%%%%%%%%% PRINT SYSTEM DETAILS %%%%%%%%%%
disp("----------------");
disp(" System Details");
disp("----------------");
disp("Pre-processing:");
disp(pre_process);
disp("Feature extraction:");
disp(feature_extraction);
disp("Classifier:");
disp(training);
disp("System Accuracy:");
disp(accuracy);
disp("----------------");


save pca_lda_svm_model model;