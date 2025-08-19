
% Load Pre-trained Network
net = densenet201;

% Load Images
rootFolder = fullfile('D:\PhD topic\Confucal dataset');
imds = imageDatastore(fullfile(rootFolder, 'N&M'), ...
    'LabelSource', 'foldernames', 'IncludeSubfolders', true, ...
    'FileExtensions', {'.jpg', '.jpeg', '.png', '.bmp', '.jfif'});

% Balancing & Counting Images w.r.t all the classes
tbl = countEachLabel(imds);
minSetCount = min(tbl{:, 2});
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Split the data into training and test sets
[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');

% Image Augmentation and Preprocessing
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, ...
    'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, ...
    'ColorPreprocessing', 'gray2rgb');

% Modify the pre-trained network for transfer learning
% Replace the final layers according to your dataset's number of classes
numClasses = numel(categories(trainingSet.Labels));
lgraph = layerGraph(net);

% Remove the final layers
lgraph = removeLayers(lgraph, {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

% Add new layers for your specific task
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')];

% Connect the new layers to the network
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'fc');

% Set training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 64, ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 1e-4, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'ValidationData', augmentedTestSet);

% Train the network
trainedNet = trainNetwork(augmentedTrainingSet, lgraph, options);

% Evaluate the trained network on the test set
predictedLabels = classify(trainedNet, augmentedTestSet);
actualLabels = testSet.Labels;

% Calculate accuracy
accuracy = mean(predictedLabels == actualLabels);
disp(['Test Accuracy: ', num2str(accuracy * 100), '%']);

% Display the confusion matrix
confMat = confusionmat(actualLabels, predictedLabels);
confMat = bsxfun(@rdivide, confMat, sum(confMat, 2));
disp('Confusion Matrix (percentage form):');
disp(confMat);

% Optionally, save the trained network and other variables if needed
save('trainedDenseNet.mat', 'trainedNet', 'confMat', 'accuracy');
