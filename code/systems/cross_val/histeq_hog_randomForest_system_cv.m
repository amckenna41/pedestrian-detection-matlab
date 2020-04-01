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
% Feature extraction: [full_image, hog_feature_vector, edgeextraction]
% Training: [svm_training, NNtraining, randomForest]
% Testing: [svm_testing, NNTesting, KNNTesting]

% assign variables for system
num_imgs = 3000;
dataset_file = 'images.dataset';
dataset_dir = '..\..\..\';

% set function handles for each stage of the system
pre_process = @enhanceContrastHE;
feature_extraction = @hog_feature_vector;
training = @randomForest;

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
    % find number of features returned by feature extraction technique
    num_feats = size(feature_extraction(images(1,:)),2);
    % apply feature extraction technique and store features to matrix
    features_train = zeros(size(images_train,1), num_feats);
    features_test = zeros(size(images_test,1), num_feats);

    for i = 1:size(images_train,1)
        features_train(i,:) = feature_extraction(images_train(i,:));
    end

    for i = 1:size(images_test,1)
        features_test(i,:) = feature_extraction(images_test(i,:));
    end

    %%%%%%%%%% TRAIN MODEL %%%%%%%%%%
    %numTrees = [25,50,100,200,250,500] -Different values of numTrees tested
    %~200 trees provided the optimum accuracy for classification
    numTrees = 200; 
    predictions = zeros(num_imgs/2, 1);


    model = training(features_train, labels_train,numTrees);
    test_model = model.predict(features_test); %testing
    predictions = str2double(test_model);
    comparison = (predictions==labels_test);
    %calculating classification accuracy on each fold 
    accuracy = sum(comparison)/length(comparison);
    tree_accuracy = [' The accuracy on fold ', num2str(fold), ' with ', num2str(numTrees) , ' trees is ', num2str(accuracy)];
    disp(tree_accuracy);

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

save hog_randomForest_cv_model model
