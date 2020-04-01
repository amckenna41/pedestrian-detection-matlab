function Lut = contrast_LS_LUT(m,c) 

Lut = zeros(1,256);

for i = 1:256
   Lut(i) = (m * (i-1)) + c;
end

Lut = uint8(Lut);

end

