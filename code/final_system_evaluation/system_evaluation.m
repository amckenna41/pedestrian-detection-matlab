%file that evaluates our final system of choice ->
%HOG & SVM
%Model is evaluated with both 50:50 & cross-validation

addpath('..\models')
dataset_file = 'images.dataset';
dataset_dir = '..\..\';

%load hog svm model
load hog_svm_model model

%load hog svm model trained on cross-validation
load hog_svm_system_cv modelcv

%load images
% add paths needed for system
addpath('..\helper_functions\');
addpath_recurse('..\pre_processing\');
addpath_recurse('..\classification\');
addpath('..\feature_extraction\hog\');

% load full dataset
disp("Loading images...");
[images, labels] = load_image_database(dataset_dir, dataset_file);

num_imgs = 3000;

%load hog function - used to obtain features
feature_extraction = @hog_feature_vector;

%load svm testing
testing = @svm_testing;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Evaluation 1: 50:50 Tested System

images_test = images((num_imgs/2)+1:num_imgs,:);
labels_test = labels((num_imgs/2)+1:num_imgs);

%first perform feature extraction
num_feats = size(feature_extraction(images(1,:)),2);

features_test = zeros(num_imgs/2, num_feats);
for i = 1:num_imgs/2
    features_test(i,:) = feature_extraction(images_test(i,:));
end


predictions = zeros(1500, 1);
disp("Testing model...");
for i = 1:(num_imgs/2)
    disp(i);
    predictions(i) = testing(features_test(i,:), model);
end

%calc accuracy
comparison = (labels_test==predictions);
accuracy = sum(comparison)/length(comparison);

[confusion, recall, precision, specificity, F1, false_alarm_rate] = evaluationMetrics(labels_test,predictions);

%accuracy = TP+TN /(P+N)
disp(['Accuracy - ', num2str((confusion(1) + confusion(2)) / sum(confusion)) ]);
disp(' ')
disp(['Confusion Matrix']);
disp('-----------------');
disp(['True Positives - ', num2str(confusion(1)) ]);
disp(['False Positives - ', num2str(confusion(3)) ]);
disp(['True Negatives - ', num2str(confusion(2)) ]);
disp(['False Negatives - ', num2str(confusion(4)) ]);
disp(' ')
disp(['Recall - ', num2str(recall)]);
disp(['Precision - ', num2str(precision)]);
disp(['Specificity - ', num2str(specificity)]);
disp(['F-measure - ', num2str(F1)]);
disp(['False Alarm Rate - ', num2str(false_alarm_rate)]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Evaluation 2: Cross-validated Tested System

%convert labels from [-1,1] to [0,1]
for i = 1:size(labels,1)
    if(labels(i,1) == -1)
        labels(i,1) = 0;
    end    
end 

indices = crossvalind('Kfold',labels,5);
cp = classperf(labels);

% record accuracy for each fold
fold_accuracies = zeros(1, 5);

for fold=1:5
    
    test = (indices == fold); 
    train = ~test;
    
    images_test = images(test,:);
    labels_test = labels(test,:);

    %%%%%%%%%% FEATURE EXTRACTION %%%%%%%%%%
    % find number of features returned by feature extraction technique
    num_feats = size(feature_extraction(images(1,:)),2);
    % apply feature extraction technique and store features to matrix
    features_test = zeros(size(images_test,1), num_feats);

    for i = 1:size(images_test,1)
        features_test(i,:) = feature_extraction(images_test(i,:));
    end

    %%%%%%%%%% TRAIN MODEL %%%%%%%%%%
    predictions = zeros(size(images_test,1), 1);
    for i = 1:size(images_test,1)
        predictions(i) = testing(features_test(i,:), modelcv);
    end

    %convert predictions from [-1,1] to [0,1]
    for i = 1:size(predictions,1)
        if(predictions(i,1) == -1)
            predictions(i,1) = 0;
        end    
    end    

    classperf(cp,predictions,test);
end

TP = cp.DiagnosticTable(1,1);
FP = cp.DiagnosticTable(1,2);
FN = cp.DiagnosticTable(2,1);
TN = cp.DiagnosticTable(2,2);

accuracy = cp.CorrectRate;
recall = cp.Sensitivity;
precision = cp.PositivePredictiveValue;
specificity = cp.Specificity; 
f1 = 2*TP/(2*TP+FP+FN);
FAR = 1 - specificity;

disp(['Accuracy - ', num2str(accuracy) ]);
disp(' ')
disp(['Confusion Matrix']);
disp('-----------------');
disp(['True Positives - ', num2str(TP) ]);
disp(['False Positives - ', num2str(FP) ]);
disp(['True Negatives - ', num2str(TN) ]);
disp(['False Negatives - ', num2str(FN) ]);
disp(' ')
disp(['Recall - ', num2str(recall)]);
disp(['Precision - ', num2str(precision)]);
disp(['Specificity - ', num2str(specificity)]);
disp(['F-measure - ', num2str(f1)]);
disp(['False Alarm Rate - ', num2str(FAR)]);
