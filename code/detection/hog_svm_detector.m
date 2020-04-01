function [boundingBoxes, confidences] = hog_svm_detector(model,image, windowSize)

    % define image height and width
    img_h = 480;
    img_w = 640;
    % define scales for multi-scale sliding window
    scales = [0.6, 2.2];
    % reshape image
    image = reshape(image, img_h, img_w);
    % define cell array to store bounding boxes
    boundingBoxes = {};
    % define counter to hold number of pedestrians found
    hits = 1;
    % define vector to hold confidence values associated with each pred
    confidences = [];
    % extract window height and width
    window_height = windowSize(1);
    window_width = windowSize(2);

    for i = 1:size(scales,2)
        
        % define height, width and new image for the current scale
        img_h_scale = scales(i)*img_h;
        img_w_scale = scales(i)*img_w;
        image_scale = imresize(image, scales(i));
        
        % iterate over every pixel in the image (fitting in window)
        for y = 1:window_height/4:(img_h_scale - window_height)
            for x = 1:window_width/4:(img_w_scale - window_width)

                % crop image to window size
                window = [x,y,window_width-1,window_height-1];
                I = imcrop(image_scale,window);
                % extract hog features for window
                features = hog_feature_vector(reshape(I,[],1));
                % get prediction from svm model
                [prediction, maxi] = svm_testing(features,model);

                if prediction == 1
                    % pedestrian found, increase counter and store bounding
                    % box plus confidence value
                    boundingBoxes{hits} =[x/scales(i), y/scales(i), window_width/scales(i), window_height/scales(i)];
                    confidences = [confidences; maxi];
                    hits = hits + 1;
                end    
            end
        end 
    end    
end