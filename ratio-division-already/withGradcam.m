
% Load the pre-trained classifier session and features
load('testFeatures53N&&T.mat');  % Test features
load('trainingfeaturesorg53N&&T.mat');  % Training features
load('N&T_classifier_session_darknet53_two_classes_for_random.mat');  % Classifier session

% Assuming your trained model is saved in the classifier session as 'net'
% If not, replace 'net' with the variable containing the trained model

% Define input size based on the model's input layer
inputSize = net.Layers(1).InputSize;

% Load a new image to test (for example, a tumor image)
sampleImage = imread('D:\Confucal dataset\random 200\200.jpg');
sampleImage = imresize(sampleImage, inputSize(1:2));  % Resize to match input size of the network

% Classify the new image using the loaded model
[YPred, scores] = classify(net, sampleImage);

% Generate Grad-CAM map for the predicted class
gradCAMMap = gradCAM(net, sampleImage, YPred);

% Resize Grad-CAM map to match the original image size
gradCAMMap = imresize(gradCAMMap, size(sampleImage, [1 2]));

% Overlay Grad-CAM on the original image
figure;
imshow(sampleImage);
hold on;
imagesc(gradCAMMap, 'AlphaData', 0.5);  % Adjust transparency of the heatmap
colormap jet;
colorbar;
title(['Grad-CAM Heatmap for Normal']);



% Optionally, repeat the above steps for another image (e.g., a normal image)
normalImage = imread('D:\Confucal dataset\random 200\26.jpg');
normalImage = imresize(normalImage, inputSize(1:2));

% Classify the normal image
[YPredNormal, ~] = classify(net, normalImage);

% Generate Grad-CAM map for the normal image
gradCAMMapNormal = gradCAM(net, normalImage, YPredNormal);
gradCAMMapNormal = imresize(gradCAMMapNormal, size(normalImage, [1 2]));

% Plot the normal image with Grad-CAM overlay
figure;
imshow(normalImage);
hold on;
imagesc(gradCAMMapNormal, 'AlphaData', 0.5);
colormap jet;
colorbar;
title(['Grad-CAM Heatmap for ', char(YPredNormal)]);






















%%%%%%%%%%%%%%%%%%%%%%%%%% for single single input
% Load the images
normalImage = imread('D:\PhD topic\Confucal dataset\presentation\heatmap_images\N.jpg');
tumorImage = imread('D:\PhD topic\Confucal dataset\presentation\heatmap_images\T1.jpg');

% Resize images (if necessary)
inputSize = [224 224];  % Example size for a pre-trained model
normalImage = imresize(normalImage, inputSize);
tumorImage = imresize(tumorImage, inputSize);

% Create imageDatastore if you have a dataset folder
imds = imageDatastore({'D:\PhD topic\Confucal dataset\presentation\heatmap_images\N.jpg', 'D:\PhD topic\Confucal dataset\presentation\heatmap_images\T1.jpg'}, 'Labels', categorical({'Normal', 'Tumor'}));
imds.ReadFcn = @(x) imresize(imread(x), inputSize);



% Define the network layers
layers = [
    imageInputLayer([224 224 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

% Training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imds, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train the network
net = trainNetwork(imds, layers, options);


% Select a sample image (e.g., the tumor image)
sampleImage = imread('D:\PhD topic\Confucal dataset\presentation\heatmap_images\T1.jpg');
sampleImage = imresize(sampleImage, inputSize);

% Classify the image and obtain scores
[YPred, scores] = classify(net, sampleImage);

% Compute Grad-CAM without specifying a layer (uses the last conv layer by default)
gradCAMMap = gradCAM(net, sampleImage, YPred);

% Resize Grad-CAM map to match the original image size
gradCAMMap = imresize(gradCAMMap, size(sampleImage, [1 2]));

% Overlay Grad-CAM map on original image
figure;
imshow(sampleImage);
hold on;
imagesc(gradCAMMap, 'AlphaData', 0.5);
colormap jet;
colorbar;
title('Grad-CAM Heatmap');





% Assuming the images have been preprocessed and resized as necessary
inputSize = [224 224];  % Example size, adjust as per your model

% Compute Grad-CAM for the Normal image
normalImage = imread('D:\PhD topic\Confucal dataset\presentation\heatmap_images\N.jpg');
normalImage = imresize(normalImage, inputSize);
normalImageCAM = gradCAM(net, normalImage, classify(net, normalImage));

% Compute Grad-CAM for the Tumor image
tumorImage = imread('D:\PhD topic\Confucal dataset\presentation\heatmap_images\T1.jpg');
tumorImage = imresize(tumorImage, inputSize);
tumorImageCAM = gradCAM(net, tumorImage, classify(net, tumorImage));

% Plotting side by side
figure;

% Plot the Normal image with Grad-CAM overlay
subplot(1,2,1);
imshow(normalImage);
hold on;
imagesc(imresize(normalImageCAM, inputSize), 'AlphaData', 0.5);
colormap jet;
colorbar;
title('');

% Plot the Tumor image with Grad-CAM overlay
subplot(1,2,2);
imshow(tumorImage);
hold on;
imagesc(imresize(tumorImageCAM, inputSize), 'AlphaData', 0.5);
colormap jet;
colorbar;
title('');




