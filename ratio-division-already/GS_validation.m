
%%%%%%%%%%%%% GS validation with N vs T (with DL and Machine leanring classifiers )
load testFeatures53N&&T.mat;  % Test features
load trainingfeaturesorg53N&&T.mat; % Training features
% load N&T_classifier_session_darknet53_two_classes_for_random.mat;


% Load Excel file containing true labels for the 200 images
filename = 'D:\PhD topic\Confucal dataset\random_200.xlsx'; % Adjust the path
opts = detectImportOptions(filename);
GS_labels = readtable(filename, opts);  % Assuming the table contains 'Number' and 'Label' columns


% Create an imageDatastore for the 200 mixed images
imdsGS = imageDatastore('D:\PhD topic\Confucal dataset\random 200', 'FileExtensions', {'.jpg','.jpeg','.png'});

% Extract features from the GS images using the pre-trained model
augmentedGSSet = augmentedImageDatastore(imageSize, imdsGS, 'ColorPreprocessing', 'gray2rgb');
GS_features = activations(net, augmentedGSSet, featureLayer, 'MiniBatchSize', 64, 'OutputAs', 'columns');

% Convert GS labels to categorical format if needed
GS_labels.Label = categorical(GS_labels.Label);  % Assuming 'Label' is the column name

% Assign labels from the Excel file to the imageDatastore
imdsGS.Labels = GS_labels.Label;

load('N&T_classifier_session_darknet53_two_classes_for_random.mat', 'SessionData');


% Predict the labels of the GS images
predictedLabels_GS = predict(classifier, GS_features, 'ObservationsIn', 'columns');

% Compare the predicted labels with the true labels from the Excel file
trueLabels_GS = GS_labels.Label;

% Confusion matrix and accuracy
confMat_GS = confusionmat(trueLabels_GS, predictedLabels_GS);
accuracy_GS = sum(predictedLabels_GS == trueLabels_GS) / numel(trueLabels_GS);

disp(['Validation Accuracy with GS data: ', num2str(accuracy_GS)]);

% Optionally, display the confusion matrix
confusionchart(trueLabels_GS, predictedLabels_GS);


% Load the pre-trained DarkNet53 network
net = darknet53;

% Load the Gold Standard (GS) Excel file containing true labels for the 200 images
filename = 'D:\PhD topic\Confucal dataset\random_200.xlsx'; % Adjust the path
opts = detectImportOptions(filename);
GS_labels = readtable(filename, opts);  % Assuming the table contains 'Number' and 'Label' columns

% Create an imageDatastore for the 200 mixed images
imdsGS = imageDatastore('D:\PhD topic\Confucal dataset\random 200', 'FileExtensions', {'.jpg','.jpeg','.png'});

% Load training and test features from previous datasets if saved
load('testFeatures53N&&T.mat');  % Test features
load('trainingfeaturesorg53N&&T.mat'); % Training features

% Pre-process images from GS using the same image size as the DarkNet53 network
imageSize = net.Layers(1).InputSize;  % Get input size of the DarkNet53 network
augmentedGSSet = augmentedImageDatastore(imageSize, imdsGS, 'ColorPreprocessing', 'gray2rgb');

% Extract Features from the GS images using the pre-trained model (DarkNet53)
featureLayer = 'avg1';  % Use 'avg_pool' as feature layer
GS_features = activations(net, augmentedGSSet, featureLayer, 'MiniBatchSize', 64, 'OutputAs', 'columns');

% Convert GS labels from Excel to categorical format if needed
GS_labels.Label = categorical(GS_labels.Label);  % Assuming 'Label' is the column name in Excel

% Assign labels from Excel file to the imageDatastore (imdsGS)
imdsGS.Labels = GS_labels.Label;

% Create the validation data as a cell array with images and labels
validationLabels = GS_labels.Label;  % Get labels from Excel file

% Create the augmented datastore for validation images
augmentedValidationSet = augmentedImageDatastore(imageSize, imdsGS, 'ColorPreprocessing', 'gray2rgb');

% Extract layers but use layerGraph to modify the network structure for transfer learning
lgraph = layerGraph(net);

% Remove the final layers from DarkNet53
lgraph = removeLayers(lgraph, {'conv53', 'softmax', 'output'});

% Define the new layers for your specific classification task
numClasses = numel(categories(trainingSet.Labels));  % Assuming 'trainingSet' is defined
newLayers = [
    fullyConnectedLayer(numClasses, 'Name', 'fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classification')];

% Add new layers and connect them to the network
lgraph = addLayers(lgraph, newLayers);
lgraph = connectLayers(lgraph, 'avg1', 'fc');

% Split the original dataset into training and validation sets
[trainingSet, validationSet] = splitEachLabel(imds, 0.8, 'randomized');  % imds is your original dataset

% Prepare the augmented image datastores for training and validation
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedValidationSet = augmentedImageDatastore(imageSize, validationSet, 'ColorPreprocessing', 'gray2rgb');

% Set the training options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {augmentedValidationSet, validationSet.Labels}, ...  % Correct validation data
    'Verbose', false, ...
    'Plots', 'training-progress');

% Fine-tune the network on the new dataset
netTransfer = trainNetwork(augmentedTrainingSet, lgraph, options);

% Evaluate the model using the validation set
predictedLabels_GS = classify(netTransfer, augmentedValidationSet);
accuracy_GS = mean(predictedLabels_GS == validationSet.Labels);

disp(['Validation Accuracy: ', num2str(accuracy_GS)]);

predictedLabels_GS = classify(netTransfer, augmentedGSSet);

% --- Compare the predicted labels with the true labels from the Excel file
trueLabels_GS = GS_labels.Label;

% --- Confusion matrix and accuracy
confMat_GS = confusionmat(trueLabels_GS, predictedLabels_GS);
accuracy_GS = sum(predictedLabels_GS == trueLabels_GS) / numel(trueLabels_GS);

disp(['Validation Accuracy with GS data: ', num2str(accuracy_GS)]);

% Optionally, display the confusion matrix
confusionchart(trueLabels_GS, predictedLabels_GS);

% --- Prepare for Classifier Learner (if you want to do further experiments with machine learning classifiers)
% Convert features to table for Classifier Learner
xGS = array2table(GS_features');
xGS.type = trueLabels_GS;  % Include labels

% Save the feature table for Classifier Learner
save('GS_features_for_ClassifierLearner.mat', 'xGS');

