%%Script that creates an SVM model using power law 
%%as a pre-processing technique and edge extraction for feaure extraction
%%and SVM as classification

clear all
close all 

%adding relevant function paths 
addpath ..\..\helper_functions\
addpath_recurse  ..\..\classification\
addpath_recurse ..\..\pre_processing

%loading 50% of training images
disp("Loading Images...");
disp("=================");
num_imgs = 3000;
[images, labels] = load_image_database('..\..\..\', 'images.dataset');

%splitting dataset into training and testing
training = images(1:(num_imgs/2),:);
training_label = labels(1:(num_imgs/2));

test = images((num_imgs/2)+1:num_imgs,:);
test_label = labels((num_imgs/2)+1:num_imgs);

[imageRow imageCol] = size(training); 
[edge_matrix] = zeros(imageRow, imageCol); 
[rowSize colSize] = size(edge_matrix); 

%gamma = 0.5;
gamma = 2; 

%brightness extraction on training images
disp("Power Law and edge extraction on Training Images...");
disp("===================================================");

for i = 1:rowSize
    
    img = reshape(training(i,:),160,96);
    img = uint8(img);
    enhanced_img = enhanceContrastPL(img, gamma); 
    [edges, Ihor, Iver] = edgeextraction(enhanced_img); 
    edge_matrix(i,:) = edges; 
    
end

%create edge extraction model from enhanced training mages
disp("Creating SVM Model...");
disp("=====================");
edge_svm_model = svm_training(edge_matrix, training_label);


[test_images_row test_images_col] = size(test);
[edge_matrix_test] = zeros(test_images_row, test_images_col);
[rowSize_test colSize_test] = size(edge_matrix_test); 

%edge extraction on training images
disp("Power Law and edge extraction on Test Images...");
disp("===============================================");

for i = 1:rowSize_test
          
    img = reshape(test(i,:),160,96);
    img = uint8(img);
    enhanced_img = enhanceContrastPL(img, gamma); 
    [edges, Ihor, Iver] = edgeextraction(enhanced_img); 
    edge_matrix_test(i,:) = edges; 
    
end


% calculate predictions from the svm model on the test images 
for i = 1:test_row
    predictions(i,1) = svm_testing(edge_matrix_test(i,:),edge_svm_model);
end

%set the negative labels to 0 prior to comparison
test_label(test_label==-1)=0;

% compare and display the accuracy of the model 
disp("SVM Classification Accuracy...");
disp("============================");
comparison = (test_label == predictions);
accuracy = sum(comparison)/length(comparison);
disp(accuracy);

%saving model 
%save powerlaw_edge_svm edge_svm_model
