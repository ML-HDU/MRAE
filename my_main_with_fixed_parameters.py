import os
import numpy
import torch
import scipy.io
from sklearn import preprocessing
from my_customs import *
from HLS_OC import *

import matplotlib
import matplotlib.pyplot as plt

# region AE algorithm
AE_type = 4
# 1: OMRAE
# 2: WSI-OMRAE
# 3: WID-OMRAE
# 4: DMRAE,
# 5: WSI-DMRAE
# 6: WID-DMRAE
flag_GPU = 1
number_of_AEs = 1
number_of_repeats = 5
# endregion

# region Dataset
name_of_dataset = 'MNIST'
flag_normalization = 1.1
tensor_size = (28, 28, -1)

evaluation_metric = 'auc'  # overall_accuracy, Gmean, R, P, F1, auc
class_chosen = 1  # , 4, 5, 6, 7, 8, 9, 10
class_chosen = numpy.array(class_chosen) - 1  # Subtract 1 because counting from 0 in Python

path_of_dataset = os.path.join('.', 'datasets', name_of_dataset)
# endregion

# region basic parameters settings

AE_parameters = numpy.array([])
HLS_OC_parameters = my_Struct()
HLS_OC_parameters.activation_function = numpy.array([None] * number_of_AEs)
HLS_OC_parameters.parameter_of_activation_function = numpy.array([None] * number_of_AEs)

for k in range(number_of_AEs):
    AE_parameters = numpy.append(AE_parameters, my_Struct())
    AE_parameters[k].flag_GPU = flag_GPU
    AE_parameters[k].activation_function = 'sigmoid'
    AE_parameters[k].parameter_of_activation_function = numpy.float64(1)
    AE_parameters[k].number_of_hidden_layer_neurons = numpy.float64(100)
    AE_parameters[k].C = 10 ** numpy.float64(0)

    HLS_OC_parameters.activation_function[k] = 'sigmoid'
    HLS_OC_parameters.parameter_of_activation_function[k] = numpy.float64(1)

HLS_OC_parameters.flag_GPU = flag_GPU
HLS_OC_parameters.C = 10 ** numpy.float64(-5)
HLS_OC_parameters.mu = numpy.float64(0.1)

# endregion


#region Determine the number of classes.
T = scipy.io.loadmat(os.path.join(path_of_dataset, 'train_label.mat'), mat_dtype=True)
T = T['T']

label = numpy.unique(T)
number_of_classes = len(label) # 确定有多少类
del T
if (-1) in label:
    # Check if there is a '-1' in label.
    # It is because the label of the anomaly class is set to be -1 in the following.
    raise SystemError('The label includes "-1"')

#endregion


#region Constructing the OCC dataset.
P = scipy.io.loadmat(os.path.join(path_of_dataset, 'train_data.mat'), mat_dtype=True)
P = P['P']

T = scipy.io.loadmat(os.path.join(path_of_dataset, 'train_label.mat'), mat_dtype=True)
T = T['T']

TV_P = scipy.io.loadmat(os.path.join(path_of_dataset, 'test_data.mat'), mat_dtype=True)
TV_P = TV_P['TV_P']

TV_T = scipy.io.loadmat(os.path.join(path_of_dataset, 'test_label.mat'), mat_dtype=True)
TV_T = TV_T['TV_T']

if flag_normalization == 0:
    print('Without Normalization')
elif flag_normalization == 1:
    P = P / 255
    TV_P = TV_P / 255
elif flag_normalization == 1.1:
    P = P * 2 - 1
    TV_P = TV_P * 2 - 1
elif flag_normalization == 2:
    ps = preprocessing.MinMaxScaler((-1,1))
    P = ps.fit_transform(P.T).T
    TV_P = ps.transform(TV_P.T).T
    del ps
elif flag_normalization == 3:
    P = ( (P - P.min()) / (P.max() - P.min()) ) * 2 - 1
    TV_P = ( (TV_P - P.min()) / (P.max() - P.min()) ) * 2 - 1
else:
    raise SystemError('Unknow switch case.')

target_class = numpy.argwhere(T == label[class_chosen])  # Positions of the targets
outlier_class = numpy.argwhere(T != label[class_chosen])  # Positions of the anomalies
T = T[target_class[:, 0]]
P = P[:, target_class[:, 0]]

P = P.reshape(tensor_size)
P = P.transpose(2, 0, 1)
del target_class
del outlier_class

target_class = numpy.argwhere(TV_T == label[class_chosen])  # Positions of the targets
outlier_class = numpy.argwhere(TV_T != label[class_chosen])  # Positions of the anomalies
TV_T[outlier_class[:, 0]] = -1

TV_P = TV_P.reshape(tensor_size)
TV_P = TV_P.transpose(2, 0, 1)
del target_class
del outlier_class

data = Dataset(P, T, TV_P, TV_T)
del P
del T
del TV_P
del TV_T
# endregion


testing_accuracy = numpy.array( [None]*number_of_repeats )
ave_testing_accuracy = my_Struct()
set_random_seed(1)
for k in range(number_of_repeats):
    _, _, _, _, _, testing_accuracy[k] = HLS_OC(data, AE_type, number_of_AEs,
                                                  AE_parameters, HLS_OC_parameters, flag_GPU)

ave_testing_accuracy.overall_accuracy = numpy.float64(0)
ave_testing_accuracy.Gmean = numpy.float64(0)
ave_testing_accuracy.R = numpy.float64(0)
ave_testing_accuracy.P = numpy.float64(0)
ave_testing_accuracy.F1 = numpy.float64(0)
ave_testing_accuracy.auc = numpy.float64(0)
for k in range(number_of_repeats):
    ave_testing_accuracy.overall_accuracy = ave_testing_accuracy.overall_accuracy + testing_accuracy[k].overall_accuracy / number_of_repeats
    ave_testing_accuracy.Gmean = ave_testing_accuracy.Gmean + testing_accuracy[k].Gmean / number_of_repeats
    ave_testing_accuracy.R = ave_testing_accuracy.R + testing_accuracy[k].R / number_of_repeats
    ave_testing_accuracy.P = ave_testing_accuracy.P + testing_accuracy[k].P / number_of_repeats
    ave_testing_accuracy.F1 = ave_testing_accuracy.F1 + testing_accuracy[k].F1 / number_of_repeats
    ave_testing_accuracy.auc = ave_testing_accuracy.auc + testing_accuracy[k].auc / number_of_repeats

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
print('Class: {}'.format(class_chosen))
print('The testing_accuracy.auc: ' + ', '.join(
    ['{:.2%}'.format(testing_accuracy[k].auc) for k in range(number_of_repeats)]))
print('The ave_testing_accuracy.auc:{:.2%}'.format(ave_testing_accuracy.auc))
