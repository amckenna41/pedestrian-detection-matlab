clear all;
close all;

% define general variables needed
num_imgs = 100;
img_h = 480;
img_w = 640;
w_size = [160, 96];
detector_box_num = 1;
dataset_file = 'detector_images.dataset';
test_dataset = 'detector_test.dataset';
dataset_dir = '..\';

% set detector to function handle
detector = @hog_svm_detector;

%load in our saved svm model
disp("Loading model...");
load models\hog_svm_model

% add any paths needed for system
addpath('helper_functions\');
addpath_recurse('pre_processing\');
addpath_recurse('classification\');
addpath_recurse('feature_extraction\');
addpath_recurse('detection\');

% define variables needed to store detection data and ground truth data
% loaded from the dataset file.
% data is stored in 2 parts, an array of boxes and an array of number of
% pedestrians per image.
disp("Loading dataset...");
detector_boxes = {};
detector_confs = [];
detector_num_peds = [];
[test_num_peds, test_boxes] = load_detector_boxes(dataset_dir, test_dataset);
[images, labels] = load_image_database(dataset_dir, dataset_file);
images = images(1:num_imgs,:);
labels = labels(1:num_imgs);

% carry out detection using detector
disp("Doing detection...");
for i = 1:num_imgs
    disp(i);
    image = images(i,:);
    [boxes, confidences] = detector(model, image, w_size);
    [boxes, confidences] = simple_nms(boxes, confidences, 0.2);  
    detector_num_peds = [detector_num_peds, size(boxes,1)];
    for j = 1:detector_num_peds(i)
        detector_boxes{detector_box_num} = boxes{j};
        detector_confs = [detector_confs, confidences(j)];
        detector_box_num = detector_box_num + 1;
    end
end

% convert detection data into correct format for evaluation
disp("Calculating results...");
detector_boxes = detector_boxes(:);
test_boxes = test_boxes(:);
temp_d_boxes = zeros(size(detector_boxes,1), 4);
temp_t_boxes = zeros(size(test_boxes,1), 4);
for i = 1:size(detector_boxes,1)
    temp_d_boxes(i,:) = detector_boxes{i};
end
for i = 1:size(test_boxes,1)
    temp_t_boxes(i,:) = test_boxes{i};
    temp_t_boxes(i,1) = temp_t_boxes(i,1)-(temp_t_boxes(i,3)/2);
    temp_t_boxes(i,2) = temp_t_boxes(i,2)-(temp_t_boxes(i,4)/2);
end
detector_boxes = temp_d_boxes;
test_boxes = temp_t_boxes;

% draw figures showing detector predictions alongside correct results,
% calculate true positive, false positive and false negatives also.
true_pos = 0;
false_pos = 0;
false_neg = 0;
detector_box_num = 1;
test_box_num = 1;

for i = 1:num_imgs
    % create matrices to hold detector & test boxes for this image
    d_boxes = zeros(detector_num_peds(i),4);
    t_boxes = zeros(test_num_peds(i),4);
    
    % first draw detector results
    figure;
    subplot(1,2,1);
    imshow(reshape(images(i,:), img_h, img_w),[]);
    hold on
    for j = 1:detector_num_peds(i)
        box = detector_boxes(detector_box_num,:);
        rectangle('Position',box,'LineWidth',1, 'EdgeColor','r');
        detector_box_num = detector_box_num + 1;
        d_boxes(j,:) = box;
    end
    drawnow();
    hold off

    % then draw correct results
    subplot(1,2,2);
    imshow(reshape(images(i,:), img_h, img_w),[]);
    hold on
    for j = 1:test_num_peds(i)
        box = test_boxes(test_box_num,:);
        rectangle('Position',box,'LineWidth',1, 'EdgeColor','g');
        test_box_num = test_box_num + 1;
        t_boxes(j,:) = box;
    end
    drawnow();
    hold off
    
    [tp, fp, fn] = evaluate_detection(d_boxes, t_boxes);
    true_pos = true_pos + tp;
    false_pos = false_pos + fp;
    false_neg = false_neg + fn;
    
end

% display results
disp("-----------------");
disp("Number of Images:");
disp(num_imgs);
disp("-----------------");
disp("Correct Number of Pedestrians:");
disp(sum(test_num_peds(1:num_imgs)));
disp("Predicted Number of Pedestrians;")
disp(sum(detector_num_peds(1:num_imgs)));
disp("-----------------");
disp("True Pos:");
disp(true_pos);
disp("False Pos:");
disp(false_pos);
disp("False Neg:");
disp(false_neg);
disp("-----------------");
disp("Precision:")
disp(true_pos / (true_pos + false_pos));
disp("Recall:");
disp(true_pos / (true_pos + false_neg));
disp("-----------------");
