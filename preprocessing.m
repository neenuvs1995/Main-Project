%% Aerial Scene Classification via Multilevel ...
%%Fusion based Deep Convolutional Neural Networks

% Initialisation
clc;
fprintf('\nInitialising ...')
clearvars;
close all;
warning('off')

showImages = 0;
showLayerGraphs = 0;
performTraining = 0;
calculateAccuracy = 0;
%% Reading and Preparing Data
fprintf('\nReading and Preparing Data ...')
dbPath = 'C:\Users\NEETHU-PC\Desktop\Hashim _aerial\neenu\AID';

pathCheck = dir(dbPath);
if isempty(pathCheck)
    errordlg('Error.Check Database Path');
    return
end

fprintf('\nData storage directory:\n\t %s',dbPath);
imds_L1 = imageDatastore(dbPath,...
    'IncludeSubfolders', true,...
    'LabelSource','foldernames',...
    'ReadFcn',@readImage_level1);

imds_L2 = imageDatastore(dbPath,...
    'IncludeSubfolders', true,...
    'LabelSource','foldernames',...
    'ReadFcn',@readImage_level2);

imds_L3 = imageDatastore(dbPath,...
    'IncludeSubfolders', true,...
    'LabelSource','foldernames',...
    'ReadFcn',@readImage_level3);

fprintf('\nSplitting train-test data');
[traindb_L1,testdb_L1] = splitEachLabel(imds_L1,0.7,'randomized');
[traindb_L2,testdb_L2] = splitEachLabel(imds_L2,0.7,'randomized');
[traindb_L3,testdb_L3] = splitEachLabel(imds_L3,0.7,'randomized');

fprintf('\nTrain set details:\n')
T = countEachLabel(traindb_L1);
labels_unique = T.Label;
disp(T)

numclasses = numel(unique(imds_L1.Labels));
fprintf('\nNumber of classes:%d',numclasses);

labels_L1 = traindb_L1.Labels;
labels_L2 = traindb_L2.Labels;
labels_L3 = traindb_L3.Labels;

%% Display a sample of training images.
if showImages
    fprintf('\nDisplaying a sample of training images ...')
    figure;
    idx = randperm(numel(traindb_L1.Files),25);
    count = 1;c = {};
    for i = idx
        c{count} = readimage(traindb_L1,i);
        count = count + 1;
    end
    montage(c,'BorderSize',[1,1],'BackgroundColor',[.3 .1 .2])
    title('Display a sample of training images')
end
