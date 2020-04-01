function [true_pos, false_pos, false_neg] = evaluate_detection(detection_boxes, correct_boxes)

true_pos = 0;
false_pos = 0;
threshold = 0.5;
found = [];

for i = 1:size(detection_boxes,1)
    
    overlapping = false;
    d_box = detection_boxes(i,:);
    
    for j = 1:size(correct_boxes,1)
        if ismember(j,found) == false
            c_box = correct_boxes(j,:);
            if intersection_over_union(d_box, c_box) > threshold
                found = [found, j];
                overlapping = true;
            end
        end
    end
    
    if overlapping == true
        true_pos = true_pos + 1;
    else
        false_pos = false_pos + 1;
    end
    
end

false_neg = size(correct_boxes,1) - size(found, 2);

end