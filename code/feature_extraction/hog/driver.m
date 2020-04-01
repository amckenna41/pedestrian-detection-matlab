% script which does hog feature extraction and svm classification for
% subset of dataset. training/testing method NOT implemented yet (50:50,
% cross-validation etc).

clear all;
close all;

addpath  ..\..\classification\svm\svm_km\
addpath  ..\..\classification\svm\

% step 1
[images, labels] = load_image_database('..\..\..\', 'images.dataset', 5);

% step 2
img = mat2gray(reshape(images(1,:),160,96));
feat = hog_feature_vector(vec2mat(images(1,:),96));

% step 3
figure;
subplot(1,2,1);
imshow(img);
subplot(1,2,2);
showHog(feat, [160 96]);

% step 4
features = zeros(100, 7524);
for i = 1:size(images,1)
    features(i,:) = hog_feature_vector(reshape(images(i,:),160,96));
end

% step 5
svm_model = svm_training(features, labels);

% step 7
[test_images, test_labels] = load_image_database('..\..\..\', 'images.dataset', 7);

predictions = zeros(1500, 1);
for i = 1:97
    predictions(i) = svm_testing(hog_feature_vector(reshape(test_images(i,:),160,96)),svm_model);
end

% step 8
comparison = (test_labels(1:97)==predictions(1:97));
accuracy = sum(comparison)/length(comparison);
disp(accuracy);

