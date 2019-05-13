function varargout = aerial_GUI(varargin)
% AERIAL_GUI MATLAB code for aerial_GUI.fig
%      AERIAL_GUI, by itself, creates a new AERIAL_GUI or raises the existing
%      singleton*.
%
%      H = AERIAL_GUI returns the handle to a new AERIAL_GUI or the handle to
%      the existing singleton*.
%
%      AERIAL_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in AERIAL_GUI.M with the given input arguments.
%
%      AERIAL_GUI('Property','Value',...) creates a new AERIAL_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before aerial_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to aerial_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help aerial_GUI

% Last Modified by GUIDE v2.5 13-May-2019 21:11:21

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @aerial_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @aerial_GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before aerial_GUI is made visible.
function aerial_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to aerial_GUI (see VARARGIN)

% Choose default command line output for aerial_GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes aerial_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = aerial_GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
  [fn,pn ] = uigetfile(fullfile('C:\Users\NEETHU-PC\Desktop\final1\neenu\AID_TEST','*.jpg'));
    img1 = fullfile(pn,fn);
    axes(handles.axes1);
    imshow(img1);title('Query image');
    handles.ImgData1 = img1;
    guidata(hObject,handles);
    
% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 clc;
 set(handles.edit1,'string','');
 I3 = handles.ImgData1;
 
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
fprintf('\nNumber of classes:%d\n',numclasses);

labels_L1 = traindb_L1.Labels;
labels_L2 = traindb_L2.Labels;
labels_L3 = traindb_L3.Labels;


%% level1
% fprintf('\nLevel_1 | vgg16:Creating network ...')
% net_L1 = vgg16;
% 
% inputSize = net_L1.Layers(1).InputSize;
% layersTransfer = net_L1.Layers(1:end-3);
% layers_L1 = [
%     layersTransfer
%     fullyConnectedLayer(numclasses,'Name','fc8','WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
%     softmaxLayer('Name','softmaxL')
%     classificationLayer('Name','classi')];
% 
% 
% 
% 
% %% level2
% 
% fprintf('\nLevel_2 | CustomNetwork:Defining Deepnet Layers ...')
% layers_L2 = [imageInputLayer([169 169 1],'Normalization','zerocenter','Name','input')
%     convolution2dLayer([7,7],96,'Stride',4,'Name','CS1')
%     reluLayer('Name','relu1')
%     maxPooling2dLayer(2,'Stride',2,'Name','PS1')
%     convolution2dLayer([5,5],256,'Stride',2,'Name','CS2')
%     reluLayer('Name','relu2')
%     maxPooling2dLayer(2,'Stride',2,'Name','PS2')
%     convolution2dLayer([3,3],512,'Stride',1,'Name','CS3')
%     reluLayer('Name','relu3')
%     fullyConnectedLayer(4096,'Name','fc4')
%     fullyConnectedLayer(4096,'Name','fc5')
%     fullyConnectedLayer(numclasses,'Name','fc6')
%     softmaxLayer('Name','softmax_L2')
%     classificationLayer('Name','final')];
% 
% 
% %% CAFFENET
% 
% addpath(genpath('hybridCNN'))
% solver_file = 'hybridCNN_deploy.prototxt';
% datafile = 'imagenet_googlenet.caffemodel';
% layers_L3 = importCaffeLayers(solver_file);
% lg = layerGraph(layers_L3);
% lg = replaceLayer(lg,'fc8',...
%     fullyConnectedLayer(numclasses,'Name','final_fcLayer'));
% lg = replaceLayer(lg,'prob',...
%     softmaxLayer('Name','softmax_L3'));
% lg = replaceLayer(lg,'ClassificationOutput',...
%     classificationLayer('Name','final'));
% 

disp('Loading saved network')
load('Level_1_vgg16.mat');
load('Level_2_custom.mat');
load('Level_3_CAFFE.mat');
load('Accuracy.mat');
Amin = min(A);
Amax = max(A);

tau = 1;
numerator = A - Amin - tau * (A - Amax);
denominator = Amax - Amin;

w = numerator ./ denominator;
networks = {net_L1,net_L2,net_L3};



networks = {net_L1,net_L2,net_L3};
fprintf('\nPerforming classification on single-image')
    
   
    index_pos = predictor_function(I3,networks,numclasses,w);
    my_pred = labels_unique(index_pos);
    my_prediction = string(my_pred);
   set(handles.edit1,'string',my_prediction);

function edit1_Callback(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of edit1 as text
%        str2double(get(hObject,'String')) returns contents of edit1 as a double


% --- Executes during object creation, after setting all properties.
function edit1_CreateFcn(hObject, eventdata, handles)
% hObject    handle to edit1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.edit1,'String','');


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
close(handles.aerial_GUI);
