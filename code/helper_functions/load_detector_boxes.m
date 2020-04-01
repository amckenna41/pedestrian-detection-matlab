function [num_peds, boxes] = load_detector_boxes(relative_dir, filename)

fp = fopen(strcat(relative_dir, filename), 'rb');
assert(fp ~= -1, ['Could not open ', strcat(relative_dir, filename), '']);

line1 = fgetl(fp);
num_imgs = str2double(fgetl(fp));

num_peds = [];
box_nums = [];

for i = 1:num_imgs
    
    line = split(fgetl(fp));
    num_peds = [num_peds, str2double(line(2))];
    
    for i = 3:size(line)-1
        if (line(i) ~= "0")
            box_nums = [box_nums, str2double(line(i))];
        end
    end
end

boxes = {};
count = 1;
box_count = 1;
for i = 1:size(num_peds,2)
    for j = 1:num_peds(i)
        b = [];
        for k = 1:4
            b = [b, box_nums(count)];
            count = count + 1;
        end
        boxes{box_count} = b;
        box_count = box_count + 1;
    end
end

fclose(fp);

end

