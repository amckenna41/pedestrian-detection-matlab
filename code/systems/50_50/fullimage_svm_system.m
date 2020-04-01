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
% Feature extraction: [full_image, hog_feature_vector, edgeextraction]
% Training: [svm_training, NNtraining]
% Testing: [svm_testing, NNTesting, KNNTesting]

% assign variables for system
num_imgs = 3000;
dataset_file = 'images.dataset';
dataset_dir = '..\..\..\';

% set function handles for each stage of the system
pre_process = false;
feature_extraction = @full_image;
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
% find number of features returned by feature extraction technique
num_feats = size(feature_extraction(images(1,:)),2);
% apply feature extraction technique and store features to matrix
features_train = zeros((num_imgs/2), num_feats);
features_test = zeros(num_imgs/2, num_feats);
for i = 1:(num_imgs/2)
    features_train(i,:) = feature_extraction(images_train(i,:));
    features_test(i,:) = feature_extraction(images_test(i,:));
end

%%%%%%%%%% TRAIN & TEST MODEL %%%%%%%%%%
disp("Training model...");
model = training(features_train, labels_train);
predictions = zeros(num_imgs/2, 1);
disp("Testing model...");
for i = 1:(num_imgs/2)
    disp(i);
    predictions(i) = testing(features_test(i,:), model);
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

save fullimage_svm_model model;