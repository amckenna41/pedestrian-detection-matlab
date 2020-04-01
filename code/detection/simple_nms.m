function [final_boxes, confs] = simple_nms(boundingBoxes, confidences, threshold)

% create structures to store data
final_boxes = {};
boxes = zeros(size(boundingBoxes,2), 5);

% combine boundingBoxes and confidences so they can be sorted
for i = 1:size(boxes)
    boundingBoxes{1,i}(5) = confidences(i);
    boxes(i,:) = boundingBoxes{1,i};
end

% sort boxes by confidences (strongest confidence = last)
boxes = sortrows(boxes, 5);

%get rid of boxes which have a confidence value of less than the threshold
cutoff = 0;
confidence_threshold = 0.5;
for i = 1:size(boxes,1)
    if boxes(i,5) >= confidence_threshold
        cutoff = i;
        break;
    end
end
if cutoff > 0
    boxes = boxes(cutoff:size(boxes,1),:);
else
    boxes = {};
end

confs = [];

% loop until boxes is empty
while ~isempty(boxes)
    
    % set current box (ind) to the last (highest confidence)
    ind = size(boxes,1);
    final_boxes = [final_boxes; boxes(ind,1:4)];
    confs = [confs, boxes(ind,5)];
    to_remove = [ind];
    
    % for every other box, if overlap area > threshold, add to remove list
    for i = 1:size(boxes,1)-1
        int_area = rectint(boxes(ind,1:4), boxes(i,1:4));
        box_area = min(boxes(ind,3)*boxes(ind,4), boxes(i,3)*boxes(i,4));
        if int_area / box_area > threshold
            to_remove = [to_remove; i];
        end
    end
    
    % remove overlapping boxes
    boxes(to_remove,:) = [];
    
end

end
