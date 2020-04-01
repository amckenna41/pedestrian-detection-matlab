function Lut = brightnessLUT(c)

Lut = zeros(1,256);

for i = 1:256
   Lut(i) = i - 1 + c;
end

Lut = uint8(Lut);

end