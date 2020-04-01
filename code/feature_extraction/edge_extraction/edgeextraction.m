function [Edges, Ihor, Iver] = edgeextraction(Iin)

% convert vector to matrix
Iin = vec2mat(uint8(Iin),96);

%Vertical edges and horizontal edges mask
B1 = [-1 0 1; -1 0 1; -1 0 1];
B2 = [-1 -1 -1; 0 0 0; 1 1 1];

% convolution on image with vertical edge mask
Ihor = edgeconvolution(Iin, B1);

% convolution on image with horizontal edge mask
Iver = edgeconvolution(Iin, B2);

[rowLen, colLen] = size(Ihor);

% intialise matrix to store extracted edges values
Edges = zeros(rowLen, colLen);

% Extract edges from horizontal and vertical image convolutions
for i=1:rowLen
    for j=1:colLen
         Edges(i,j) = sqrt((Iver(i,j)^2) + ((Ihor(i,j)^2)));
    end
end

%pad edge array with zeros due to loss of information from convolution 
Edges = padarray(Edges, [1 1]); 

% get row and column size of image
[rowLen, colLen] = size(Edges);

%reshape edges 2D array back into 1 dimensional array 
Edges = reshape(Edges, [1, (rowLen) * (colLen)]);

end
