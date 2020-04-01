%randomForest function takes in the training images and labels and the
%number of decision trees to be created in the Random Forest. Treebagger
%function is then called on images, labels and tree number
function TreeBag = randomForest(trainImages, trainLabels, numTrees)

%Treebagger class bags an ensemble of decision trees for classification
TreeBag = TreeBagger(numTrees, trainImages, trainLabels,'OOBPrediction', 'On','Method', 'classification');

end


