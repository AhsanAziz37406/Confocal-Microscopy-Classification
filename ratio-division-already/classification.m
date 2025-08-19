% Set up the image data store
imageData = 'D:\Confugal dataset\CRS_Data_Aug'; % Replace with your dataset path


% Load image dataset
imds = imageDatastore(imageData, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Display the number of images in each category
labelCount = countEachLabel(imds);
disp(labelCount);

% Split dataset into training and validation sets
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

% Load pre-trained ResNet-50 network
net = resnet50;

% Get the input size of the network
inputSize = net.Layers(1).InputSize(1:2);

% Resize the images to the input size of the network
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation);

% Modify the network to match the number of classes in your dataset
% Get the number of classes
numClasses = numel(categories(imdsTrain.Labels));

% Remove the last three layers
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

% Add new layers
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_classoutput')];

% Add and connect the new layers
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'new_fc');

% Set training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
netTransfer = trainNetwork(augimdsTrain, lgraph, options);

% Evaluate the network
YPred = classify(netTransfer, augimdsValidation);
YValidation = imdsValidation.Labels;

accuracy = mean(YPred == YValidation);
disp(['Validation accuracy: ', num2str(accuracy)]);


%%%%%%%%%%%%%%%%% two class


% Set up the image data store
imageData = 'D:\Confugal dataset\CRS_Data_Aug'; % Replace with your dataset path

% Load image dataset
imds = imageDatastore(imageData, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Display the number of images in each category
labelCount = countEachLabel(imds);
disp(labelCount);

% Split dataset into training and validation sets
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

% Load pre-trained ResNet-50 network
net = resnet50;

% Modify the network to match the number of classes in your dataset
% Get the number of classes
numClasses = numel(categories(imdsTrain.Labels));

% Remove the last three layers
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

% Add new layers
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'new_softmax')
    classificationLayer('Name', 'new_classoutput')];

% Add and connect the new layers
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg_pool', 'new_fc');

% Resize the images to the input size of the network
inputSize = net.Layers(1).InputSize(1:2);

% Create augmented image datastores to resize the images
augimdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize, imdsValidation);

% Set training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
netTransfer = trainNetwork(augimdsTrain, lgraph, options);

% Evaluate the network
YPred = classify(netTransfer, augimdsValidation);
YValidation = imdsValidation.Labels;

accuracy = mean(YPred == YValidation);
disp(['Validation accuracy: ', num2str(accuracy)]);


clear all;
net= densenet201;


rootFolder = fullfile('D:\Confugal dataset');

imds = imageDatastore(fullfile(rootFolder,'CRS_Data_Aug'), 'LabelSource','foldernames', 'IncludeSubfolders',true, 'FileExtensions',{'.jpg','.jpeg','.png','.bmp','.jfif'});
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});
imds = splitEachLabel(imds, minSetCount, 'randomize');
[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');
imageSize = net.Layers(1).InputSize
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');
%%
trainingLabels = trainingSet.Labels;
testLabels = testSet.Labels;

%%
%   Extract Training Features Using CNN
featureLayer = 'avg_pool';

tic
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
save testFeaturesdense.mat
feat=testFeatures';
feat=double(feat);
label=testLabels;
label=double(label);
ho = 0.2; 
% Hold-out method
HO = cvpartition(label,'HoldOut',ho,'Stratify',false);

% Parameter setting
N        = 10; 
max_Iter = 20; 
tau      = 1; 
eta      = 1; 
alpha    = 1; 
beta     = 1; 
rho      = 0.2; 
phi      = 0.5; 
Nf       = 1000;       % Set number of selected features
% Ant Colony System
[sFeat,Nf,Sf,curve] = jACO(feat,label,N,max_Iter,tau,eta,alpha,beta,rho,phi,Nf,HO);

% Plot convergence curve
plot(1:max_Iter,curve); 
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('ACS'); grid on;


x1=sFeat;
y=testLabels;

x = array2table(x1);
x.type = y;


%%
%   Train A Multiclass SVM Classifier Using CNN Features
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');


predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

confMat = confusionmat(testLabels, predictedLabels);

confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

mean(diag(confMat))
toc


x = array2table(testFeatures');
x.type = testLabels;
classificationLearner

testfeaturesorgg=x;
save testfeaturesorgdense.mat


x = array2table(trainingFeatures');
x.type = trainingLabels;

trainingfeaturesorgg=x;
save trainingfeaturesorgdense.mat


classificationLearner

