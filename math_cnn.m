function [net, info] = math_cnn(varargin)
% Here activated pixels from the Leap device are treated as a binary 0/1

run(fullfile('C:\Users\Henry\Box Sync\Projects\matconvnet-master\matlab', ...
    'vl_setupnn.m')) ;

opts.dataDir = fullfile('data');
opts.expDir = fullfile('data', 'math');
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 200 ;
opts.train.numEpochs = 200 ;  
opts.train.continue = true ;  % can continue training after stopping
opts.train.useGpu = true ;
opts.train.learningRate = [0.1*ones(1, 50) 0.01*ones(1, 50) 0.001*ones(1, 50) 0.0001*ones(1,50)] ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.90 ;
opts.train.outputClasses = 101 ;
opts.train.expDir = opts.expDir ;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Define a network similar to LeNet
f= 1/100;
net.layers = {};

f=1/100 ;
net.layers = {} ;
% data is 31x31
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(4,4,1,20, 'single'), ...
                           'biases', zeros(1, 20, 'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,5,20,50, 'single'),...
                           'biases', zeros(1,50,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(5,5,50,500, 'single'),...
                           'biases', zeros(1,500,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', ...
                           'filters', f*randn(1,1,500,opts.train.outputClasses, 'single'),...
                           'biases', zeros(1,opts.train.outputClasses ,'single'), ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
tic ; 
% Take the mean out and make GPU if needed
if opts.train.useGpu
  imdb.images.data = gpuArray(imdb.images.data) ;
end

[net, info] = cnn_train(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

toc
end 
% --------------------------------------------------------------------
function [im, labels] = getBatch(imdb, batch)
% --------------------------------------------------------------------
im = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
end 
% --------------------------------------------------------------------
function imdb = getImdb(opts)
% --------------------------------------------------------------------

% Prepare the imdb structure, returns image data with mean image subtracted
if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

trainfrac = 0.7;

% get all the directories & remove '.' & '..'
files = dir('images');
fileNames = {files(~[files.isdir]).name};

index = 1;
for i=1:length(fileNames),
  image = load(fullfile('images', fileNames{i}));
  % imshow(image);  
  images(:,:,index) = image;
  % this is the output class, 1 through 12directoryNames{i}
  output(index) = split(fileNames{i});       
  index = index + 1;
  if mod(index, 1000) ==  0,
      sprintf('Parsed %d', index)
  end;
end;

% mix up the classes 
shuffle = randperm(size(output,2));
images = images(:,:,shuffle);
output = output(shuffle);
% split into train and test
trainsize = int64(trainfrac*size(output,2));
testsize = size(output,2) - trainsize;

% set is a row of ones then threes used by library for training and test sets
% a two would be validation set, not used here
set = [ones(1,trainsize) 3*ones(1,testsize)];
% added a space for convolutions
data = single(reshape(images,size(images,1),size(images,2),1,[]));

% get the mean of each image so it can be subtracted
% and divide by the std dev
dataMean = mean(data(:,:,:,set == 1), 4);
dataStd = std(data(:,:,:,set == 1), 0, 4);
data = bsxfun(@minus, data, dataMean) ;
data = bsxfun(@rdivide, data, dataStd) ;

imdb.images.data = data;
imdb.images.data_mean = dataMean;
imdb.images.labels = output;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'};
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),1:5,'uniformoutput',false); 
end 

% --------------------------------------------------------------------
function class = split(filename)
% --------------------------------------------------------------------

% Find the delimiters
delimIdx = find(filename == '_');
% Find the text between the delimiters
% don't include the delimiters
startOffset = delimIdx(end)+1;
delimIdx = find(filename == '.');
endOffset = delimIdx-1;
% Get the element
txt = filename(startOffset:endOffset);
% Attempt conversion to number
class = sscanf(txt, '%f');

end
