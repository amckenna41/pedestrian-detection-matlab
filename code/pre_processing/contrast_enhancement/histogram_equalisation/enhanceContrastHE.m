function Iout = enhanceContrastHE(Iin)

% convert input image from vector into uint8 image
Iin = vec2mat(uint8(Iin),96);

%Call histogram equalisation look-up table function 
Lut = contrast_HE_LUT(Iin); 

%Convert input image to grayscale 
if (length(size(Iin)) == 3)
    Iin = rgb2gray(Iin);
end

%Output eqalised image 
Iout = intlut(Iin, Lut);

% convert input image back into vector
Iout = reshape(Iout,1,160*96);

end

