%---Used GPU automatically if cuda-toolkit and cudNN lib installed with microsoft VS 2015

clear all;
net = darknet53;

rootFolder = fullfile('D:\Confucal dataset\confocal');
imds = imageDatastore(fullfile(rootFolder,'N&T(all)'), 'LabelSource','foldernames', 'IncludeSubfolders',true, 'FileExtensions',{'.jpg','.jpeg','.png','.bmp','.jfif'});

%   Balancing & Counting Images w.r.t all the classes
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});
imds = splitEachLabel(imds, minSetCount, 'randomize');

%%
%   Prepare Training and Test Image Sets
[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');

%%
%   Pre-process Images For CNN
imageSize = net.Layers(1).InputSize
% imageSize = net.meta.normalization.imageSize(1:3)
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

%%
%   Prepare Training and Test Labels
trainingLabels = trainingSet.Labels;
testLabels = testSet.Labels;


%% Extract Training and Testing Features Using CNN
featureLayer = 'avg1';

tic
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
save testFeatures53N&&T.mat

%% Feature Selection Using ACO
feat = testFeatures';
feat = double(feat);
label = testLabels;
label = double(label);
ho = 0.2; % Hold-out method
HO = cvpartition(label, 'HoldOut', ho, 'Stratify', false);

% Parameter setting
N = 10; 
max_Iter = 20; 
tau = 1; 
eta = 1; 
alpha = 1; 
beta = 1; 
rho = 0.2; 
phi = 0.5; 
Nf = 1200; % Set number of selected features

% Ant Colony System
[sFeat, Nf, Sf, curve] = jACO(feat, label, N, max_Iter, tau, eta, alpha, beta, rho, phi, Nf, HO);

% Plot convergence curve
plot(1:max_Iter, curve); 
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('ACS');
grid on;

%% Train a Multiclass SVM Classifier Using CNN Features
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

%% Evaluate Classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide, confMat, sum(confMat, 2));

% Display the mean accuracy
mean(diag(confMat))
toc

%% Prepare feature_Vector(x) for classification
x = array2table(testFeatures');
x.type = testLabels;

testfeaturesorgg = x;
save testfeaturesorg53N&&T.mat

x = array2table(trainingFeatures');
x.type = trainingLabels;

trainingfeaturesorgg = x;
save trainingfeaturesorg53N&&T.mat

classificationLearner
