function [IoU] = intersection_over_union(box1, box2)

union = box1(3)*box1(4) + box2(3)*box2(4) - rectint(box1,box2);
IoU = rectint(box1,box2) / union;

end

