import os
import numpy
import scipy.io
import torch
from my_customs import *
import math
import time
from MRAE import *
import matplotlib.pyplot as plt


def HLS_OC(data, AE_type, number_of_AEs, AE_parameters, HLS_OC_parameters, flag_GPU):
    # region Import training data
    P = data.P
    T = data.T.T

    label = numpy.unique(T)
    T = numpy.ones(T.shape)

    # endregion

    # region Autoencoder
    training_begin_time = time.time()
    for k in range(number_of_AEs):

        if AE_type == 1:  # OMRAE
            AE_beta = numpy.array([None] * number_of_AEs)
            AE_beta[k] = OMRAE(P, flag_GPU,
                               AE_parameters[k].number_of_hidden_layer_neurons, AE_parameters[k].C,
                               AE_parameters[k].activation_function,
                               AE_parameters[k].parameter_of_activation_function)
            P = AE_beta[k] @ P

        elif AE_type == 2:  # WSI-OMRAE
            AE_beta = numpy.array([None] * number_of_AEs)
            AE_beta[k] = WSI_OMRAE(P, flag_GPU,
                               AE_parameters[k].number_of_hidden_layer_neurons, AE_parameters[k].C,
                               AE_parameters[k].activation_function,
                               AE_parameters[k].parameter_of_activation_function)
            P = AE_beta[k] @ P

        elif AE_type == 3:  # WID-OMRAE
            AE_beta = numpy.array([None] * number_of_AEs)
            AE_beta[k] = WID_OMRAE(P, flag_GPU,
                                   AE_parameters[k].number_of_hidden_layer_neurons, AE_parameters[k].C,
                                   AE_parameters[k].activation_function,
                                   AE_parameters[k].parameter_of_activation_function)
            P = AE_beta[k] @ P

        elif AE_type == 4:  # DMRAE
            AE_beta_left = numpy.array([None] * number_of_AEs)
            AE_beta_right = numpy.array([None] * number_of_AEs)
            AE_beta_left[k] = OMRAE(P, flag_GPU,
                                    AE_parameters[k].number_of_hidden_layer_neurons, AE_parameters[k].C,
                                    AE_parameters[k].activation_function,
                                    AE_parameters[k].parameter_of_activation_function)
            AE_beta_right[k] = OMRAE(P.transpose(0, 2, 1), flag_GPU,
                                     AE_parameters[k].number_of_hidden_layer_neurons, AE_parameters[k].C,
                                     AE_parameters[k].activation_function,
                                     AE_parameters[k].parameter_of_activation_function)
            P = AE_beta_left[k] @ P @ AE_beta_right[k].T

        elif AE_type == 5:  # WSI-DMRAE
            AE_beta_left = numpy.array([None] * number_of_AEs)
            AE_beta_right = numpy.array([None] * number_of_AEs)
            AE_beta_left[k] = WSI_OMRAE(P, flag_GPU,
                                        AE_parameters[k].number_of_hidden_layer_neurons, AE_parameters[k].C,
                                        AE_parameters[k].activation_function,
                                        AE_parameters[k].parameter_of_activation_function)
            AE_beta_right[k] = WSI_OMRAE(P.transpose(0, 2, 1), flag_GPU,
                                         AE_parameters[k].number_of_hidden_layer_neurons, AE_parameters[k].C,
                                         AE_parameters[k].activation_function,
                                         AE_parameters[k].parameter_of_activation_function)
            P = AE_beta_left[k] @ P @ AE_beta_right[k].T

        elif AE_type == 6:  # WID-DMRAE
            AE_beta_left = numpy.array([None] * number_of_AEs)
            AE_beta_right = numpy.array([None] * number_of_AEs)
            AE_beta_left[k] = WID_OMRAE(P, flag_GPU,
                                        AE_parameters[k].number_of_hidden_layer_neurons, AE_parameters[k].C,
                                        AE_parameters[k].activation_function,
                                        AE_parameters[k].parameter_of_activation_function)
            AE_beta_right[k] = WID_OMRAE(P.transpose(0, 2, 1), flag_GPU,
                                         AE_parameters[k].number_of_hidden_layer_neurons, AE_parameters[k].C,
                                         AE_parameters[k].activation_function,
                                         AE_parameters[k].parameter_of_activation_function)
            P = AE_beta_left[k] @ P @ AE_beta_right[k].T

        else:
            raise SystemError('Unknow switch case.')

        if HLS_OC_parameters.activation_function[k].lower() == 'sigmoid':
            P = 1 / (1 + numpy.exp(-HLS_OC_parameters.parameter_of_activation_function[k] * P))
        elif HLS_OC_parameters.activation_function[k].lower() == 'scaling sigmoid':
            scaling_parameter = HLS_OC_parameters.parameter_of_activation_function[k] / numpy.abs(P).max()
            P = 1 / (1 + numpy.exp(-scaling_parameter * P))
        elif HLS_OC_parameters.activation_function[k].lower() == 'sine':
            P = numpy.sin(P)
        elif HLS_OC_parameters.activation_function[k].lower() == 'linear':
            pass
        else:
            raise SystemExit('Unknown activation function.')

    # endregion

    # region Train HLS-OC
    P = P.reshape(P.shape[0], -1).T

    number_of_hidden_layer_neurons, number_of_training_data = P.shape

    H = P.T
    del P
    T = T.T

    if flag_GPU == 1:
        H = torch.from_numpy(H).cuda()
        T = torch.from_numpy(T).cuda()

        if number_of_training_data >= number_of_hidden_layer_neurons:
            Hsquare = torch.matmul(H.T, H)
            HLS_OC_beta = torch.linalg.solve(
                torch.eye(H.shape[1], dtype=torch.float64, device='cuda') * (1 / HLS_OC_parameters.C) + Hsquare,
                torch.matmul(H.T, T))
        else:
            Hsquare = torch.matmul(H, H.T)
            HLS_OC_beta = torch.matmul(H.T, torch.linalg.solve(
                torch.eye(H.shape[0], dtype=torch.float64, device='cuda') * (1 / HLS_OC_parameters.C) + Hsquare, T))

        HLS_OC_beta = HLS_OC_beta.cpu().numpy()
        H = H.cpu().numpy()
        T = T.cpu().numpy()

    else:

        if number_of_training_data >= number_of_hidden_layer_neurons:
            Hsquare = numpy.dot(H.T, H)
            HLS_OC_beta = numpy.linalg.solve(numpy.eye(H.shape[1]) * (1 / HLS_OC_parameters.C) + Hsquare,
                                             numpy.dot(H.T, T))
        else:
            Hsquare = numpy.dot(H, H.T)
            HLS_OC_beta = numpy.dot(H.T,
                                    numpy.linalg.solve(numpy.eye(H.shape[0]) * (1 / HLS_OC_parameters.C) + Hsquare, T))

    del Hsquare

    # endregion

    # region Obtain the training time and compute the output of training data计算训练时间，训练数据的实际输出
    training_time = time.time() - training_begin_time

    Y = numpy.dot(H, HLS_OC_beta)
    del H

    # endregion

    # region Compute the threshold

    distance_error = numpy.abs(Y - T)
    distance_error = numpy.abs(numpy.sort(-distance_error, axis=0))

    theta = distance_error[int(numpy.floor(number_of_training_data * HLS_OC_parameters.mu))]

    del distance_error
    # endregion

    # region Obtain the overall accuracy of training data
    distance_error = numpy.abs(Y - T)
    distance_error = distance_error.flatten()
    MissClassificationRate_Training = numpy.float64(0)

    for i in range(number_of_training_data):
        if distance_error[i] >= theta:
            MissClassificationRate_Training = MissClassificationRate_Training + 1

    training_accuracy = 1 - MissClassificationRate_Training / number_of_training_data

    del Y
    del T
    del distance_error
    # endregion

    # region Compute the output of the testing data
    TV_P = data.TV_P
    TV_T = data.TV_T.T
    number_of_testing_data = TV_T.shape[1]
    number_of_testing_data_outlier_class = numpy.argwhere(TV_T == -1).shape[0]
    number_of_testing_data_target_class = number_of_testing_data - number_of_testing_data_outlier_class

    testing_begin_time = time.time()  # 开始计时

    for k in range(number_of_AEs):
        if AE_type in (1, 2, 3):
            TV_P = AE_beta[k] @ TV_P
        elif AE_type in (4, 5, 6):
            TV_P = AE_beta_left[k] @ TV_P @ AE_beta_right[k].T
        else:
            raise SystemError('Unknow switch case.')

        if HLS_OC_parameters.activation_function[k].lower() == 'sigmoid':
            TV_P = 1 / (1 + numpy.exp(-HLS_OC_parameters.parameter_of_activation_function[k] * TV_P))
        elif HLS_OC_parameters.activation_function[k].lower() == 'sine':
            TV_P = numpy.sin(TV_P)
        elif HLS_OC_parameters.activation_function[k].lower() == 'linear':
            pass
        else:
            raise SystemExit('Unknown activation function.')

    TV_P = TV_P.reshape(TV_P.shape[0], -1).T

    H_test = TV_P.T
    del TV_P

    TY = numpy.dot(H_test, HLS_OC_beta).T
    testing_time = time.time() - testing_begin_time
    del H_test
    del HLS_OC_beta

    # endregion

    # region testing_accuracy.auc
    testing_accuracy = my_Struct()

    distance_error = numpy.abs(TY - numpy.ones(TV_T.shape))  # 这时候distance_error是一个行向量
    testing_accuracy.auc, _, _ = my_roc_curve(distance_error, TV_T, label, 2)
    # endregion

    # region Conduct decision
    TY = TY.flatten()
    TV_T = TV_T.flatten()
    distance_error = distance_error.flatten()
    for k in range(number_of_testing_data):
        if distance_error[k] >= theta:
            TY[k] = -1
        else:
            TY[k] = label

    # endregion

    # region testing_accuracy.overall_accuracy
    MissClassificationRate_Testing = numpy.float64(0)
    for i in range(number_of_testing_data):
        if TV_T[i] != TY[i]:
            MissClassificationRate_Testing = MissClassificationRate_Testing + 1

    testing_accuracy.overall_accuracy = 1 - MissClassificationRate_Testing / number_of_testing_data

    # endregion

    # region testing_accuracy.Gmean
    testing_accuracy.Gmean = numpy.float64(1)
    label_including_netative = numpy.append(label, -1)
    number_of_each_class_of_test_data = numpy.array(
        [number_of_testing_data_target_class, number_of_testing_data_outlier_class])
    for k in range(label_including_netative.size):
        position = numpy.argwhere(
            TV_T == label_including_netative[k])
        if position.size != 0:
            testing_accuracy.Gmean = testing_accuracy.Gmean * (
                    numpy.sum(TY[position] == label_including_netative[k]) / number_of_each_class_of_test_data[k])

    testing_accuracy.Gmean = testing_accuracy.Gmean ** (1 / label_including_netative.size)

    # endregion

    # region Recall，testing_accuracy.R
    position_positive = numpy.argwhere(TV_T == label)
    if position_positive.size != 0:
        testing_accuracy.R = numpy.sum(TY[position_positive] == label) / number_of_testing_data_target_class

    # endregion

    # region Precision，testing_accuracy.P
    position_positive = numpy.argwhere(TV_T == label)
    position_negative = numpy.argwhere(TV_T != label)
    if position_positive.size != 0 and position_negative.size != 0:
        testing_accuracy.P = numpy.sum(TY[position_positive] == label) / (
                numpy.sum(TY[position_positive] == label) + numpy.sum(TY[position_negative] == label))

    # endregion

    # region F1,testing_accuracy.F1
    testing_accuracy.F1 = (2 * testing_accuracy.P * testing_accuracy.R) / (testing_accuracy.P + testing_accuracy.R)

    # endregion

    # print('Pause')

    return TY, distance_error, training_time, training_accuracy, testing_time, testing_accuracy
