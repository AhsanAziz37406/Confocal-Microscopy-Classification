% Load the trained model
load testFeatures53N&&T.mat; % Replace with actual model if necessary
load trainingfeaturesorg53N&&T.mat; % Load the training features if needed
load N&T_classifier_session_darknet53_two_classes_for_random.mat; % Assuming you have saved the classifier in a separate file

% Define source folder containing 200 mixed images and destination folder
srcFolder = 'D:\Confucal dataset\random 200';  % Folder with random mixed images
destFolder = 'D:\Confucal dataset\Random_two_cases'; % Folder to save renamed images

% Create destination folder if it doesn't exist
if ~exist(destFolder, 'dir')
    mkdir(destFolder);
end

% Create an imageDatastore for the mixed images
imdsMixed = imageDatastore(srcFolder, 'FileExtensions', {'.jpg', '.jpeg', '.png', '.bmp', '.jfif'});

% Augment and resize images to match input size of the network
imageSize = net.Layers(1).InputSize; % Input size of Darknet53
augmentedMixedSet = augmentedImageDatastore(imageSize, imdsMixed, 'ColorPreprocessing', 'gray2rgb');

% Extract features from mixed images using the trained network
mixedFeatures = activations(net, augmentedMixedSet, featureLayer, 'MiniBatchSize', 64, 'OutputAs', 'columns');

% Predict labels for the mixed images
predictedLabels = predict(classifier, mixedFeatures, 'ObservationsIn', 'columns');

% Loop through each image in the datastore and save with the predicted class name
for i = 1:numel(imdsMixed.Files)
    % Get the predicted label for the current image
    predictedLabel = string(predictedLabels(i));
    
    % Get the original image file name
    [~, fileName, ext] = fileparts(imdsMixed.Files{i});
    
    % Create a new name with the predicted class and numbering
    newFileName = sprintf('%s_%d%s', predictedLabel, i, ext);
    
    % Define the destination path
    destFile = fullfile(destFolder, newFileName);
    
    % Copy the image to the destination folder with the new name
    copyfile(imdsMixed.Files{i}, destFile);
    
    % Optionally, display progress
    fprintf('Image %d renamed to %s and saved.\n', i, newFileName);
end

disp('All images labeled and saved.');
