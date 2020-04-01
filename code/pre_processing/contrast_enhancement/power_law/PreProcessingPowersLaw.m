%first read in files from 'pos' folder
img_folder = dir('C:\\Users\\Nathan\\OneDrive - Queen''s University Belfast\\VAML Project\\images\\pos\\*.jpg')
addpath('C:\\Users\\Nathan\\OneDrive - Queen''s University Belfast\\VAML Project\\images\\pos\\')
files={img_folder.name}
for k=1:numel(files)
  images{k}=imread(files{k});
end
pos_size = size(images,2);

%then read in files from 'neg' folder
img_folder = dir('C:\\Users\\Nathan\\OneDrive - Queen''s University Belfast\\VAML Project\\images\\neg\\*.jpg')
addpath('C:\\Users\\Nathan\\OneDrive - Queen''s University Belfast\\VAML Project\\images\\neg\\')
files={img_folder.name};
i = pos_size;
for k=1:numel(files)
  images{i+k}=imread(files{k});
end

%shuffle images
images(randperm(length(images)));

%display 4 random images
%BEFORE PROCESSING
figure()
subplot(2,2,1)
imshow(cell2mat(images(1,80)))

subplot(2,2,2)
imshow(cell2mat(images(1,520)))

subplot(2,2,3)
imshow(cell2mat(images(1,300)))

subplot(2,2,4)
imshow(cell2mat(images(1,22)))

%attempt to apply Power's Law - varying gamma values
%1 - gamma value 0.5

for i=1:size(images,2)
    image_i = cell2mat(images(1,i));
    image_i = enhanceContrastPL(image_i, 0.5);
    images(1,i) = mat2cell(image_i, size(image_i,1), size(image_i,2), size(image_i,3));
end

%display 4 same images after enhancement 
%AFTER PROCESSING
figure()
subplot(2,2,1)
imshow(cell2mat(images(1,80)))

subplot(2,2,2)
imshow(cell2mat(images(1,520)))

subplot(2,2,3)
imshow(cell2mat(images(1,300)))

subplot(2,2,4)
imshow(cell2mat(images(1,22)))
