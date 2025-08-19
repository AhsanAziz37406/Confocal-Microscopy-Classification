% Set up the image data store
imageData = 'D:\PhD topic\Confucal dataset\CRS_Data_Aug'; % Replace with your dataset path

imds = imageDatastore(imageData, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Split dataset into training and validation sets
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');


% Create a bag of visual words from the training set
bag = bagOfFeatures(imdsTrain);

% Encode the images using the bag of visual words
trainFeatures = encode(bag, imdsTrain);
validationFeatures = encode(bag, imdsValidation);

% Get the labels for the training and validation sets
trainLabels = imdsTrain.Labels;
validationLabels = imdsValidation.Labels;



% Train an SVM classifier using the extracted features
classifier = fitcecoc(trainFeatures, trainLabels);

% Save the trained classifier to a variable
trainedClassifier = classifier;
