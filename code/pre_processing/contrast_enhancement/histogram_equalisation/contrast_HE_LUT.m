function Lut = contrast_HE_LUT(Iin)

%Create look-up table of zeros 
Lut = zeros(1,256);

%Get histogram of input image 
hist_Iin = histogram(Iin);
%Set number of bins to 256 
hist_Iin.NumBins = 256; 
%hist.BinLimits = [0, 256];

%Get values from histogram 
hist_vals = hist_Iin.Values;
%Get cumulative sum of histogram values 
CH = cumsum(double(hist_vals));

%Get total number of image pixels 
[rowSize,colSize] = size(Iin);
total_pix = rowSize * colSize; 

%Calculate new histogram values using transfer function and populate LUT 
for i = 1:256
    Lut(i) = max(0, (256*(CH(i)/(total_pix)) -1));
    
end

%Convert LUT into uint8 
Lut = uint8(Lut);

end