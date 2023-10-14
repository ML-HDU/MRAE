close all;
clear;
clc;

P = loadMNISTImages('train-images.idx3-ubyte');

TV_P = loadMNISTImages('t10k-images.idx3-ubyte');

T = loadMNISTLabels('train-labels.idx1-ubyte');
T = T + 1;

TV_T = loadMNISTLabels('t10k-labels.idx1-ubyte');
TV_T = TV_T + 1;

save(fullfile('.','train_data.mat') , 'P')
save(fullfile('.','test_data.mat') , 'TV_P')
save(fullfile('.','train_label.mat') , 'T')
save(fullfile('.','test_label.mat') , 'TV_T')