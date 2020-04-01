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
% Training: [svm_training, NNtraining, randomForest]
% Testing: [svm_testing, NNTesting, KNNTesting]

% assign variables for system
num_imgs = 3000;
dataset_file = 'images.dataset';
dataset_dir = '..\..\..\';

% set function handles for each stage of the system
pre_process = false;
feature_extraction = @hog_feature_vector;
training = @randomForest;

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
% numTrees = 200; 
numTrees = [25,50,100,200,250,500];

prediction = zeros(num_imgs/2, 6);

%Model trained, tested and accuracy calculated 
%for each value of trees in the numTrees vector 
for i=1:size(numTrees,2)

    model = training(features_train, labels_train,numTrees(1,i));
    test_model = model.predict(features_test);
    prediction(:,i) = str2double(test_model);
    comparison = (prediction(:,i)==labels_test);
    accuracy = sum(comparison)/length(comparison);
    tree_accuracy = [' The accuracy with ', num2str(numTrees(1,i)) , ' trees is ', num2str(accuracy)];
    disp(tree_accuracy);

    %figure shows the Out-Of-Bag misclassification error from the model 
    figure;
    oobErrorBaggedEnsemble = oobError(model);
    plot(oobErrorBaggedEnsemble,...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','b')
    xlabel 'Number of trees';
    ylabel 'Out-of-bag classification error';
    title 'OOB Classification error for 0 - 500 trees'

end


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

save hog_randomForest_model model;
