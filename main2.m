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
multiTest = 0;
%% Reading and Preparing Data
fprintf('\nReading and Preparing Data ...')
dbPath = 'C:\Users\NEETHU-PC\Desktop\final1\neenu\AID';

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
%% level1
fprintf('\nLevel_1 | vgg16:Creating network ...')
net_L1 = vgg16;

inputSize = net_L1.Layers(1).InputSize;
layersTransfer = net_L1.Layers(1:end-3);
layers_L1 = [
    layersTransfer
    fullyConnectedLayer(numclasses,'Name','fc8','BiasLearnRateFactor',20)
    softmaxLayer('Name','softmaxL')
    classificationLayer('Name','classi')];


lgraph = layerGraph(layers_L1);
if showLayerGraphs
    fprintf('\nLevel_1 | vgg16:Plotting layerGraph ...')
    figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
    plot(lgraph);
end
fprintf('\nLevel_1 | vgg16:Defining network properties...')

options = trainingOptions('sgdm', ...
    'MiniBatchSize',8, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

if performTraining
    tic
    fprintf('\nLevel_1 | vgg16:Training the CNN network...')
    net_L1 = trainNetwork(traindb_L1,layers_L1,options);
    save Level_1_vgg16.mat net_L1
    t1 = toc;
    fprintf('\nLevel_1 | vgg16:Time taken for training:%f secs',t1)
    
    fprintf('\nLevel_1 | vgg16:Extracting CNN features ...')
    layer = 'softmaxL';
    tic
    featuresTrain_L1 = activations(net_L1,traindb_L1,layer,'OutputAs','rows');
    t2 = toc;
    fprintf('\nLevel_1 | vgg16:Time taken for train set feature extraction:%f secs',t2)
    
    fprintf('\nLevel_1 | vgg16:Saving CNN features ...')
    save Level_1_feat.mat featuresTrain_L1 labels_L1
    
else
    fprintf('\nLevel_1 | vgg16:Loading trained network...')
    load('Level_1_vgg16.mat')
    
    fprintf('\nLevel_1 | vgg16:Loading CNN features ...')
    load('Level_1_feat.mat')
end


%% level2

fprintf('\nLevel_2 | CustomNetwork:Defining Deepnet Layers ...')
layers_L2 = [imageInputLayer([169 169 1],'Name','input')
    convolution2dLayer([7,7],96,'Stride',4,'Name','CS1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','PS1')
    convolution2dLayer([5,5],256,'Stride',2,'Name','CS2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','PS2')
    convolution2dLayer([3,3],512,'Stride',1,'Name','CS3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(4096,'Name','fc4')
    fullyConnectedLayer(4096,'Name','fc5')
    fullyConnectedLayer(numclasses,'Name','fc6')
    softmaxLayer('Name','softmax_L2')
    classificationLayer('Name','final')];

lgraph = layerGraph(layers_L2);
if showLayerGraphs
    figure
    plot(lgraph)
end
fprintf('\nLevel_2 | CustomNetwork:Specifying Training Options...')
options = trainingOptions('adam',...
    'Plots','training-progress',...
    'InitialLearnRate',0.001,...
    'MaxEpochs',50,...
    'Shuffle','every-epoch',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.0001,...
    'LearnRateDropPeriod',60,...
    'Verbose',true,...
    'VerboseFrequency',60 );

if performTraining
    fprintf('\nLevel_2 | CustomNetwork:Training the network ...\n')
    
    tic% to note down time
    net_L2 = trainNetwork(traindb_L2,layers_L2,options);
    fprintf('\nLevel_2 | CustomNetwork:Training Completed.')
    
    fprintf('\nLevel_2 | CustomNetwork:Saving trained network for future use ...')
    save Level_2_custom.mat net_L2
    
    t1 = toc;
    timeString = datestr(t1/(24*60*60), 'DD:HH:MM:SS');
    fprintf('\nLevel_2 | CustomNetwork:Time taken for training : %s',timeString)
    
    
    tic
    layer = 'softmax_L2';
    featuresTrain_L2 = activations(net_L2,traindb_L2,layer,'OutputAs','rows');
    t2 = toc;
    timeString = datestr(t2/(24*60*60), 'DD:HH:MM:SS');
    fprintf('\nLevel_2 | CustomNetwork:Time taken for train set feature extraction:%s',timeString)
    fprintf('\nLevel_2 | CustomNetwork:Saving CNN features ...')
    save Level_2_feat.mat featuresTrain_L2 labels_L2
else
    fprintf('\nLevel_2 | CustomNetwork:Loading trained network...')
    load('Level_2_custom.mat')
    
    fprintf('\nLevel_2 | CustomNetwork:Loading CNN features ...')
    load('Level_2_feat.mat')
end


%% CAFFENET

addpath(genpath('hybridCNN'))
solver_file = 'hybridCNN_deploy.prototxt';
datafile = 'imagenet_googlenet.caffemodel';
layers_L3 = importCaffeLayers(solver_file);

lg = layerGraph(layers_L3);
if showLayerGraphs
    figure
    plot(lg)
    title('CAFFE Layer graph')
end

lg = replaceLayer(lg,'fc8',...
    fullyConnectedLayer(numclasses,'Name','final_fcLayer'));
lg = replaceLayer(lg,'prob',...
    softmaxLayer('Name','softmax_L3'));
lg = replaceLayer(lg,'ClassificationOutput',...
    classificationLayer('Name','final'));

fprintf('\nLevel_3 | CAFFE:Specifying Training Options...')
options = trainingOptions('adam',...
    'Plots','training-progress',...
    'InitialLearnRate',0.0001,...
    'MaxEpochs',50,...
    'Shuffle','every-epoch',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.0001,...
    'LearnRateDropPeriod',60,...
    'Verbose',true,...
    'VerboseFrequency',60 );

if performTraining
    fprintf('\nLevel_3 | CAFFE:Training the network ...\n')
    
    tic% to note down time
    net_L3 = trainNetwork(traindb_L3,lg,options);
    fprintf('\nLevel_3 | CAFFE:Training Completed.')
    
    fprintf('\nLevel_3 | CAFFE:Saving trained network for future use ...')
    save Level_3_CAFFE.mat net_L3 lg
    
    t1 = toc;
    timeString = datestr(t1/(24*60*60), 'DD:HH:MM:SS');
    fprintf('\nLevel_3 | CAFFE:Time taken for training : %s',timeString)
    
    
    tic
    layer = 'softmax_L3';
    featuresTrain_L3 = activations(net_L3,...
        traindb_L3,layer,'OutputAs','rows');
    t2 = toc;
    timeString = datestr(t2/(24*60*60), 'DD:HH:MM:SS');
    fprintf('\nLevel_3 | CAFFE:Time taken for train set feature extraction:%s',timeString)
    fprintf('\nLevel_3 | CAFFE:Saving CNN features ...')
    save Level_3_feat.mat featuresTrain_L3 labels_L3
else
    fprintf('\nLevel_3 | CAFFE:Loading trained network...')
    load('Level_3_CAFFE.mat')
    
    fprintf('\nLevel_3 | CAFFE:Loading CNN features ...')
    load('Level_3_feat.mat')
end
%% Weight Estimation
fprintf('\n Performing Weight Estimation...')

% %Level1
% fprintf('\n\tLevel1')

fprintf('\nCreating individual datastore for each classes');
[ds_L1,ds_L2,ds_L3] = deal(cell(1,numclasses));
for i = 1:numclasses
    lbl = string(labels_unique(i));
    ds_L1{i} = splitEachLabel(traindb_L1,0.999999,'Include',lbl);
    ds_L2{i} = splitEachLabel(traindb_L2,0.999999,'Include',lbl);
    ds_L3{i} = splitEachLabel(traindb_L3,0.999999,'Include',lbl);
end

if calculateAccuracy
    
    fprintf('\nCalculating accuracy for each class for each level...')
    [A1, A2, A3] = deal(zeros(1,numclasses));
    for i = 1:numclasses
        fprintf('\n\t\t*Class%d',i)
        Y_Pred_L1 = classify(net_L1,ds_L1{i});
        Y_train_L1 = ds_L1{i}.Labels;
        A1(i) = mean(Y_Pred_L1 == Y_train_L1);
        
        Y_Pred_L2 = classify(net_L2,ds_L2{i});
        Y_train_L2 = ds_L2{i}.Labels;
        A2(i) = mean(Y_Pred_L2 == Y_train_L2);
        
        Y_Pred_L3 = classify(net_L3,ds_L3{i});
        Y_train_L3 = ds_L3{i}.Labels;
        A3(i) = mean(Y_Pred_L3 == Y_train_L3);
    end
    
    
    A = [A1;
        A2;
        A3];
    save Accuracy.mat A
else
    load Accuracy.mat
end

% Equation3
Amin = min(A);
Amax = max(A);

tau = 1;
numerator = A - Amin - tau * (A - Amax);
denominator = Amax - Amin;

w = numerator ./ denominator;
networks = {net_L1,net_L2,net_L3};

if ~multiTest
    %% Perform classi on single image
    fprintf('\nPerforming classification on single-image')
    
    % fn = 'airport_12.jpg';
    [fn,pn ] = uigetfile(fullfile('C:\Users\NEETHU-PC\Desktop\final1\neenu\AID_TEST','*.jpg'));
    fn = fullfile(pn,fn);
    index_pos = predictor_function(fn,networks,numclasses,w);
    my_pred = labels_unique(index_pos);
    my_prediction = string(my_pred);
    msgbox(my_prediction,'result')
else
    fprintf('\nPerforming classification on multiple-images\n')
    y_test = testdb_L1.Labels;
    count = 0;
    filenames = testdb_L1.Files;
    numTest = numel(filenames);
    fprintf('\nNumber of test images:%d',numTest);
    fprintf('\n\n\tACTUAL\tPREDICTED\n')
    y1 = cell(numTest,1);
    for v = 1:numTest
        
        fprintf('%d ',v)
        fn =  filenames{v};
        index_pos = predictor_function(fn,networks,numclasses,w);
        my_pred = labels_unique(index_pos);
        my_prediction = string(my_pred);
        my_actual = string(y_test(v));
        fprintf('%s\t%s\n',my_actual,my_prediction)
        %         y1{v,1} = my_prediction;
        if strcmp(my_actual,my_prediction)
            count = count+1;
        end
    end
    
    acc = count/numTest*100;
    fprintf('\nAccuracy:\n\t%f',acc)
    
    %     y_test = testdb_L1.Labels;
    
    %     strcmp(y_test,y1)
    
end