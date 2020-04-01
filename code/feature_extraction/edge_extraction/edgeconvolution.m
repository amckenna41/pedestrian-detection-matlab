function Iout = edgeconvolution(Iin,B)

% calculate image size 
[imageRows, imageCols] = size(Iin);

% calculate mask size 
[matrixRows, matrixCols] = size(B);

rowSize = imageRows - matrixRows + 1;
colSize = imageCols - matrixCols + 1;

% initialise matrix to hold convoluted image 
Iout = zeros(rowSize, colSize);
M = matrixRows;
N = matrixCols;

% set image and mask to double so convolution can be calculated 
Iin = double(Iin);
B=double(B);

% loop through image and matrix rows and columns and multiply the weights
% of the mask on the image pixels, store convolution new image. 
for k=1:rowSize
    for l=1:colSize
        for i=k:k+M-1
            for j=l:l+N-1 
                Iout(k, l) = Iout(k, l) + Iin(i,j) * B(i - k + 1, j - l + 1);
            end
        end
    end
end


