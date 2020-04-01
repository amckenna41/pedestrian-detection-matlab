function prediction = KNNTesting(testImage,modelNN)

% to fit in with our system - manually set K here rather than as parameter
K = 13;

%euclidean distance between test sample and all training sample
distances = [];
for i=1:size(modelNN.neighbours,1)
    dist = EuclideanDistance(testImage, modelNN.neighbours(i,:));
    distances = [distances;dist];
end

dist_and_labels = [distances modelNN.labels];

dist_and_labels = sortrows(dist_and_labels,1);
labels = dist_and_labels(:,2);
KNN = labels(1:K);

prediction = mode(KNN);

end

