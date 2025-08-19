%---Used GPU automatically if cuda-toolkit and cudNN lib installed with microsoft VS 2015

clear all;
net= inceptionv3;

%%
%   Load Images
rootFolder = fullfile('D:\PhD topic\braindataset');
imds = imageDatastore(fullfile(rootFolder,'brain'), 'LabelSource','foldernames', 'IncludeSubfolders',true, 'FileExtensions',{'.jpg','.jpeg','.png','.bmp','.jfif'});

%%
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

%%
%   Extract Training Features Using CNN
featureLayer = 'avg_pool';

tic
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
testFeatures = activations(net, augmentedTestSet, featureLayer, ...
    'MiniBatchSize', 64, 'OutputAs', 'columns');
save testFeaturesdenseN&M.mat
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
Nf       = 1200;       % Set number of selected features
% Ant Colony System
[sFeat,Nf,Sf,curve] = jACO(feat,label,N,max_Iter,tau,eta,alpha,beta,rho,phi,Nf,HO);

% Plot convergence curve
plot(1:max_Iter,curve); 
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('ACS'); grid on;


%%
%   Preparing feature_Vector(x) for classification
x1=sFeat;
y=testLabels;

x = array2table(x1);
x.type = y;


%%
%   Train A Multiclass SVM Classifier Using CNN Features
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

%%
%   Evaluate Classifier
predictedLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

% Display the mean accuracy
mean(diag(confMat))
toc


%   Preparing feature_Vector(x) for classification
x = array2table(testFeatures');
x.type = testLabels;
classificationLearner

testfeaturesorgg=x;
save testfeaturesorgdenseN&M.mat


x = array2table(trainingFeatures');
x.type = trainingLabels;

trainingfeaturesorgg=x;
save trainingfeaturesorgdenseN&M.mat


classificationLearner



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Clear all previous data
clear all;

% Load DenseNet-201 model
net = densenet201;

% Load Images
rootFolder = fullfile('D:\PhD topic\Confucal dataset');
imds = imageDatastore(fullfile(rootFolder, 'N&M'), 'LabelSource', 'foldernames', 'IncludeSubfolders', true, 'FileExtensions', {'.jpg', '.jpeg', '.png', '.bmp', '.jfif'});

% Balance & Count Images w.r.t all the classes
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Prepare Training and Test Image Sets
[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');

% Pre-process Images For CNN
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

% Extract Training Features Using CNN
featureLayer = 'avg_pool';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, 'MiniBatchSize', 64, 'OutputAs', 'columns');
testFeatures = activations(net, augmentedTestSet, featureLayer, 'MiniBatchSize', 64, 'OutputAs', 'columns');

% Save features and labels
save testFeaturesdenseN&M.mat testFeatures;
save trainingFeaturesdenseN&M.mat trainingFeatures;

% Load one image from the test set to generate an attention map
imageIndex = 1; % Select the first image for the example
selectedImage = readimage(testSet, imageIndex);
actualLabel = testSet.Labels(imageIndex);

% Resize the selected image to match the network's input size
resizedImage = imresize(selectedImage, imageSize(1:2)); % Resize to [224 224]

% Display the selected image
figure;
imshow(selectedImage);
title(['Selected Image - Actual Label: ', char(actualLabel)]);


% Perform prediction using the resized image
augmentedImage = augmentedImageDatastore(imageSize, resizedImage, 'ColorPreprocessing', 'gray2rgb');
predictedScores = predict(net, augmentedImage);
[~, classIdx] = max(predictedScores);

% Generate Grad-CAM for the resized image using the default last convolutional layer
% Note: No 'LayerName' parameter is passed here
scoreMap = gradCAM(net, resizedImage, classIdx);

% Resize scoreMap to match the original resized image size
scoreMap = imresize(scoreMap, [size(resizedImage, 1), size(resizedImage, 2)]);


% Overlay the attention map on the original image
figure;
imshow(selectedImage);
hold on;
imagesc(scoreMap, 'AlphaData', 0.5);
colormap jet;
colorbar;
title(['Grad-CAM - Class: ', char(actualLabel)]);
hold off;







%%%%%%%%%%%%%%%%%%%%%%%%%%



% Clear all previous data
clear all;

% Load DenseNet-201 model
net = densenet201;

% Load Images
rootFolder = fullfile('D:\PhD topic\Confucal dataset');
imds = imageDatastore(fullfile(rootFolder, 'N&M'), 'LabelSource', 'foldernames', 'IncludeSubfolders', true, 'FileExtensions', {'.jpg', '.jpeg', '.png', '.bmp', '.jfif'});

% Balance & Count Images w.r.t all the classes
tbl = countEachLabel(imds);
minSetCount = min(tbl{:,2});
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Prepare Training and Test Image Sets
[trainingSet, testSet] = splitEachLabel(imds, 0.8, 'randomize');

% Pre-process Images For CNN
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

% Extract Training Features Using CNN
featureLayer = 'avg_pool';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, 'MiniBatchSize', 64, 'OutputAs', 'columns');
testFeatures = activations(net, augmentedTestSet, featureLayer, 'MiniBatchSize', 64, 'OutputAs', 'columns');

% Save features and labels
save testFeaturesdenseN&M.mat testFeatures;
save trainingFeaturesdenseN&M.mat trainingFeatures;

% Load one image from the test set to generate an attention map
imageIndex = 1; % Select the first image for the example
selectedImage = readimage(testSet, imageIndex);
actualLabel = testSet.Labels(imageIndex);

% Resize the selected image to match the network's input size
resizedImage = imresize(selectedImage, imageSize(1:2)); % Resize to [224 224]

% Perform prediction using the resized image
augmentedImage = augmentedImageDatastore(imageSize, resizedImage, 'ColorPreprocessing', 'gray2rgb');
predictedScores = predict(net, augmentedImage);
[~, classIdx] = max(predictedScores);

% Generate Grad-CAM for the resized image using the default last convolutional layer
scoreMap = gradCAM(net, resizedImage, classIdx);

% Resize scoreMap to match the original image size
scoreMap = imresize(scoreMap, size(selectedImage, [1 2])); % Resize to [original height, original width]

% Normalize scoreMap to enhance visibility
scoreMap = mat2gray(scoreMap);

% Display the original image with full-size Grad-CAM heatmap overlay
figure;
imshow(selectedImage);
hold on;
imagesc(scoreMap, 'AlphaData', 0.5);
colormap jet;
colorbar;
title(['Grad-CAM - Class: ', char(actualLabel)]);
hold off;


%%%%%%%%%%%%%%%%%%%%%% selected image of your own choice

% Clear all previous data
clear all;

% Load DenseNet-201 model
net = densenet201;

% Specify the full paths of the images you want to use
imagePath1 = 'D:\PhD topic\Confucal dataset\presentation\M.jpg';  % Change to your image path
imagePath2 = 'D:\PhD topic\Confucal dataset\presentation\N.jpg';  % Change to your image path

% Read the specified images
selectedImage1 = imread(imagePath1);
selectedImage2 = imread(imagePath2);

% Combine selected images into a cell array for processing
selectedImages = {selectedImage1, selectedImage2};

% Prepare to store results
results = cell(size(selectedImages));

% Process each selected image
for i = 1:length(selectedImages)
    originalImage = selectedImages{i};
    
    % Resize the image to the input size expected by the network
    resizedImage = imresize(originalImage, net.Layers(1).InputSize(1:2));
    
    % Use augmented image datastore to apply preprocessing if needed
    augmentedImage = augmentedImageDatastore(net.Layers(1).InputSize, resizedImage, 'ColorPreprocessing', 'gray2rgb');
    
    % Predict using the network
    predictedScores = predict(net, augmentedImage);
    [~, classIdx] = max(predictedScores);
    
    % Generate Grad-CAM for the resized image using the default last convolutional layer
    scoreMap = gradCAM(net, resizedImage, classIdx);
    
    % Resize scoreMap to match the original image size
    scoreMap = imresize(scoreMap, size(originalImage, [1 2])); % Resize to [original height, original width]
    
    % Normalize scoreMap to enhance visibility
    scoreMap = mat2gray(scoreMap);
    
    % Store the results for each image
    results{i} = scoreMap;
    
    % Display the original image with full-size Grad-CAM heatmap overlay
    figure;
    imshow(originalImage);
    hold on;
    imagesc(scoreMap, 'AlphaData', 0.5);
    colormap jet;
    colorbar;
    title(['Grad-CAM - Image ', num2str(i)]);
    hold off;
end
