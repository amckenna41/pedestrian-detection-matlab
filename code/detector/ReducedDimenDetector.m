function [boundingBoxes] = ReducedDimenDetector(model,image, windowSize)
    
    %cell array will store bounding boxes - boundingBoxes
    % ^ 1 row for every window in a image
    %window size (x,y)
    %for multi scale, wrap in outer for loop with varying sizes of
    %coefficent .... i.e. z*height and z*width z=(0.5, 1, 3)
    
    img_w = 160;
    img_h = 96;
    
    %first image needs reshaped into 2d
    image = reshape(image, img_w, img_h);
    
    %to store bounding boxes
    boundingBoxes = {};
    
    %to store num predictions
    hits = 0;
    
    %to implement multi-scale window, 
    %current window dimensions will be multiplied by varying coefficents
    %ii = [0.5, 0.8, 1, 1.1];
    %for i = 1:4
        %z = ii(i);
        z=1;
        window_height = z*windowSize(1);
        window_width = z*windowSize(2);
        image_height = 160;
        image_width = 96;

        %for every window
        %increments by 10
        for y = 1: 5 :(image_height - window_height)
            for x = 1: 5:(image_width - window_width)

                %define current window
                window = [x,y,window_width,window_height];

                %crop image
                I = imcrop(image,window);

                %reshape to size of image that model was trained on
                %...96x160
                I = imresize(I, [img_w, img_h]);

                %before passing to HOG, requires transformation into 1D vector
                I = reshape(I,[],1);

                %generate HOG features
                features = hog_feature_vector(I);

                prediction = svm_testing(features,model);

                %store window dimensions if image in current window is predicted 
                %to be a person or not
                if prediction == 1
                    hits = hits + 1;
                    boundingBoxes{hits} =[x, y, x+window_width, y+window_height];
                end    
            end    
        end    
    %end
        %cell array of bounding boxes is returned
    
end