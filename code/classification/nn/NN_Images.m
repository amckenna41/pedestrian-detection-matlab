clear all
close all

addpath svm\svm_km\
addpath ..\feature_extraction\hog
addpath ..\

%Load in a sample of images to train model
[images, labels] = load_image_database('..\..\', 'images.dataset', 10);

%Code to split images into 50:50 sample
cv = cvpartition(size(images,1),'HoldOut',0.5);
idx = cv.test;
imgTrain = images(~idx,:);
imgTest  = images(idx,:);
lblTrain = labels(~idx,:);
lblTest = labels(idx,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Approach 1
disp(' ')
disp('Nearest Neighbour - Full Image')
disp('-------------------')

modelNN = NNtraining(imgTrain, lblTrain);

%Test model with test set of images
for i=1:size(imgTest,1)
    
    testnumber= imgTest(i,:);
    
    classificationResult(i,1) = NNTesting(testnumber, modelNN);
    
end

%Evaluate Accuracy
comparison = (lblTest==classificationResult);
Accuracy = sum(comparison)/length(comparison);

disp(['Accuracy = ', num2str(Accuracy)]);
[recall, precision, specificity, F1, false_alarm_rate] = evaluationMetrics(lblTest, classificationResult);
disp(['Recall = ', num2str(recall)]); 
disp(['Precision = ', num2str(precision)]);
disp(['Specificity = ', num2str(specificity)]);
disp(['F-measure = ', num2str(F1)]);
disp(['False Alarm Rate = ', num2str(false_alarm_rate)]);
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Approach 2

%Run HOG feature extraction
disp(' ')
disp('Nearest Neighbour - HOG Features')
disp('-------------------')

%test transformation of images from vector to matrix as approptiate
img = mat2gray(reshape(imgTrain(1,:),160,96));
feat = hog_feature_vector(reshape(imgTrain(1,:),160,96));

%evidence of HOG working
figure;
subplot(1,2,1);
imshow(img);
subplot(1,2,2);
showHog(feat, [160 96]);

%Run Hog
features = zeros(size(imgTrain,1), size(feat, 2));
for i = 1:size(imgTrain,1)
    features(i,:) = hog_feature_vector(reshape(imgTrain(i,:),160,96));
end


%Build KNN model with HOG features & labels
modelNN_hog = NNtraining(features, lblTrain);

%Test model with features and correspsonding labels
test_feature = zeros(size(imgTest,1), size(feat, 2));
for i=1:size(imgTest,1)
    
    test_feature(i,:) = hog_feature_vector(reshape(imgTest(i,:),160,96));
    
    %test model with test images' hog features
    hog_NN_classification(i,1) = NNTesting(test_feature(i,:), modelNN_hog);
    
end

%Evaluate Accuracy
hog_NN_comparison = (lblTest==hog_NN_classification);
hog_NN_accuracy = sum(hog_NN_comparison)/length(hog_NN_comparison);

disp(['Accuracy = ', num2str(hog_NN_accuracy)]);
[recall, precision, specificity, F1, false_alarm_rate] = evaluationMetrics(lblTest, hog_NN_classification);
disp(['Recall = ', num2str(recall)]); 
disp(['Precision = ', num2str(precision)]);
disp(['Specificity = ', num2str(specificity)]);
disp(['F-measure = ', num2str(F1)]);
disp(['False Alarm Rate = ', num2str(false_alarm_rate)]);
disp(' ')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Approach 3

%Perfoming KNN - comparing against the 5 Nearest Neighbours
disp(' ')
disp('KNN - FULL IMAGE')
disp('-------------------')

%Test model with test set of images
 for k = [3 5 7]
    
    for i=1:size(imgTest,1)

        testnumber= imgTest(i,:);

        classificationResult(i,1) = KNNTesting(testnumber, modelNN, k);

    end

    %Evaluate Accuracy
    knn_comparison = (lblTest==classificationResult);
    knn_accuracy = sum(knn_comparison)/length(knn_comparison);
    
    disp(['Accuracy = ', num2str(knn_accuracy), ' with k = ', num2str(k)] );
    [recall, precision, specificity, F1, false_alarm_rate] = evaluationMetrics(lblTest, classificationResult);
    disp(['Recall = ', num2str(recall)]); 
    disp(['Precision = ', num2str(precision)]);
    disp(['Specificity = ', num2str(specificity)]);
    disp(['F-measure = ', num2str(F1)]);
    disp(['False Alarm Rate = ', num2str(false_alarm_rate)]);
    disp(' ')
 end
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Approach 4

%Peforming KNN on Hog Features

disp(' ')
disp('KNN - HOG Features')
disp('-------------------')
 for k = [3 5 7]   
    test_feature = zeros(size(imgTest,1), size(feat, 2));
    for i=1:size(imgTest,1)
        test_feature(i,:) = hog_feature_vector(reshape(imgTest(i,:),160,96));

        %test model with test images' hog features
        hog_KNN_classification(i,1) = KNNTesting(test_feature(i,:), modelNN_hog, k);

    end

    %Evaluate Accuracy
    hog_KNN_comparison = (lblTest==hog_KNN_classification);
    hog_KNN_accuracy = sum(hog_KNN_comparison)/length(hog_KNN_comparison);
    
    disp(['Accuracy = ', num2str(hog_KNN_accuracy), ' with k = ', num2str(k)] ); 
    [recall, precision, specificity, F1, false_alarm_rate] = evaluationMetrics(lblTest, hog_KNN_classification);
    disp(['Recall = ', num2str(recall)]); 
    disp(['Precision = ', num2str(precision)]);
    disp(['Specificity = ', num2str(specificity)]);
    disp(['F-measure = ', num2str(F1)]);
    disp(['False Alarm Rate = ', num2str(false_alarm_rate)]);
    disp(' ')
 end


 
 
;