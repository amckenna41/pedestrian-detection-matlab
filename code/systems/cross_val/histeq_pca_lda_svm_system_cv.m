clear all;
close all;

% this script is a full machine learning pipeline which does pre
% processing, feature extraction, training and testing, and outputs an 
% accuracy %age for the system. The training/testing method used is 50:50. 
% TODO: Update this to crossvalidation.

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
pre_process = @enhanceContrastHE;
feature_extraction = @PrincipalComponentAnalysis;
training = @svm_training;
testing = @svm_testing;

% add paths needed for system
addpath('..\..\helper_functions\');
addpath_recurse('..\..\pre_processing\');
addpath_recurse('..\..\classification\');
addpath_recurse('..\..\feature_extraction\');

% load full dataset
[images, labels] = load_image_database(dataset_dir, dataset_file);

%convert labels from [-1,1] to [0,1]
for i = 1:size(labels,1)
    if(labels(i,1) == -1)
        labels(i,1) = 0;
    end    
end    

%%%%%%%%%% PRE-PROCESSING %%%%%%%%%%
% apply data pre-processing technique and store output images
if ~isequal(pre_process, false)
    for i = 1:num_imgs
        images(i,:) = pre_process(images(i,:));
    end
end

% split data into folds
tic
indices = crossvalind('Kfold',labels,5);
cp = classperf(labels);

% record accuracy for each fold
fold_accuracies = zeros(1, 5);

for fold=1:5
    
    test = (indices == fold); 
    train = ~test;
    images_train = images(train,:);
    labels_train = labels(train,:);
    images_test = images(test,:);
    labels_test = labels(test,:);

    %%%%%%%%%% FEATURE EXTRACTION %%%%%%%%%%
    %%%%%%%%%% FEATURE EXTRACTION %%%%%%%%%%
    %pc = [2,5,10,15,20,40,50]; 
    %optimal principal components number is 40 
    pc = 40; 

    disp("Extracting features...");
    %extracting features into PCA feature space
    [eigenVectors_train, eigenValues_train, meanX_train, Xpca_train] = feature_extraction(images_train, pc);
    %extracting PCA reduced features into LDA feature space
    [eigenVectors_lda, eigenValues_lda, meanX_lda, Xpca_lda] = LDA(labels_train, [], Xpca_train); 
    

    %%%%%%%%%% TRAIN MODEL %%%%%%%%%%
    disp("Training model...");
    model = training(Xpca_lda, labels_train);
    predictions = zeros(size(images_test,1), 1);
    disp("Testing model...");
    
    for i = 1:size(images_test,1)
        disp(i);
        testNumber = images_test(i,:);
        pca = (testNumber - meanX_train)*eigenVectors_train;
        lda = (pca-meanX_lda)*eigenVectors_lda;
        result = testing(lda, model);
        predictions(i) = result; 
    end
    
        
    %convert predictions from [-1,1] to [0,1]
    for i = 1:size(predictions,1)
        if(predictions(i,1) == -1)
            predictions(i,1) = 0;
        end    
    end    

    classperf(cp,predictions,test);
    
    fold_accuracies(1,fold) = cp.LastCorrectRate;
    
end

disp(fold_accuracies);
accuracy = cp.CorrectRate;
toc

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

save histeq_pca_lda_svm_cv_model model