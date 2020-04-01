function [images, labels] = load_image_database(relative_dir, filename, sampling)

if nargin<3
    sampling =1;
end

fp = fopen(strcat(relative_dir, filename), 'rb');
assert(fp ~= -1, ['Could not open ', strcat(relative_dir, filename), '']);

line1=fgetl(fp);

numberOfImages = fscanf(fp,'%d',1);

images=[];
labels =[];

for im=1:sampling:numberOfImages
    label = fscanf(fp,'%d',1);
    labels = [labels; label];
    
    imfile = fscanf(fp,'%s',1);
    I=imread(strcat(relative_dir, imfile));
    if size(I,3)>1
        I=rgb2gray(I);
    end
    
    vector = reshape(I,1, size(I, 1) * size(I, 2));
    vector = double(vector); % / 255;
    
    images= [images; vector];
    
end

fclose(fp);

end