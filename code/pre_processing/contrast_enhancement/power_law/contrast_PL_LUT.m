function Lut = contrast_PL_LUT(gamma)
 Lut = [0:255];
 for i = 1:256
  Lut(i) = power(i,gamma) / power(255, gamma-1);
 end

 Lut=uint8(Lut);
end

