function prediction = NNTesting(testImage,modelNN)
    
    %euclidean distance between test sample and all training sample
    
    min = EuclideanDistance(testImage, modelNN.neighbours(1,:));
    index = 1;
    for i=2:size(modelNN.neighbours,1)
        dist = EuclideanDistance(testImage, modelNN.neighbours(i,:));
        
        if dist < min
            min = dist;
            index = i;
        end
    end
    
    prediction = modelNN.labels(index,:);
    
end

