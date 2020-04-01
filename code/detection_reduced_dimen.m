clear all;
close all;

num_imgs = 300;
img_w = 160;
img_h = 96;
dataset_file = 'images_mac.dataset';
dataset_dir = '../';

% add any paths needed for system
addpath('helper_functions/');
addpath_recurse('pre_processing/');
addpath_recurse('classification/');
addpath_recurse('feature_extraction/');
addpath_recurse('detector/');

%load in our saved svm model
%load hog_svm_model
load histequil_dimenreduct_svm 

%load images
[images, labels] = load_image_database(dataset_dir, dataset_file, 10);

test_img = images(1,:);

%define window size
w_size = [100, 35];

boxes = ReducedDimenDetector(svm_model, test_img, w_size );

%show image
imshow(reshape(test_img, img_w, img_h),[]);

for i = 1:numel(boxes)
    box = boxes{i};
    rectangle('Position',[box(1,1),box(1,2),box(1,3)-box(1, 1),box(1,4) - box(1,2)],'LineWidth',2, 'EdgeColor','r');
end    