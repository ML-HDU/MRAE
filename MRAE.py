import os
import numpy
import scipy.io
import scipy.linalg
import torch
import math
import time

from my_customs import *


def OMRAE(P, flag_GPU, L, C, activation_function, parameter_of_activation_function):
    number_of_training_data, d1, d2 = numpy.float64(P.shape)

    # region Randomly generate the input weight and bias.
    input_weight = numpy.random.rand(int(L), int(d1)) * 2 - 1
    bias = numpy.random.rand(int(L), int(d2)) * 2 - 1
    # endregion

    # region Compute hidden-layer output
    tempH = input_weight @ P + bias
    del input_weight
    del bias

    if activation_function.lower() == 'sigmoid':
        H = 1 / (1 + numpy.exp(-parameter_of_activation_function * tempH))
    elif activation_function.lower() == 'scaling sigmoid':
        scaling_parameter = parameter_of_activation_function / numpy.abs(tempH).max()
        H = 1 / (1 + numpy.exp(-scaling_parameter * tempH))
    elif activation_function.lower() == 'sine':
        H = numpy.sin(tempH)
    elif activation_function.lower() == 'linear':
        H = tempH
    else:
        raise SystemError('Unknown activation function.')

    del tempH

    # endregion

    # region Derive the output weight.
    H = H.transpose(0, 2, 1)

    if flag_GPU == 1:
        # raise SystemError('We think the analytical solution is not suitable for using GPU. '
        #                   'The corresponding implement is thus not completed.')
        H = torch.from_numpy(H).cuda()
        P = torch.from_numpy(P).cuda()

        Hsquare = torch.sum(torch.matmul(H.permute(0, 2, 1), H), dim=0)

        HTXT = torch.sum(torch.matmul(H.permute(0, 2, 1), P.permute(0, 2, 1)), dim=0)

        beta = torch.linalg.solve(torch.eye(Hsquare.shape[0], dtype=torch.float64, device='cuda') * (1 / C) + Hsquare,
                                  HTXT)

        beta = beta.cpu().numpy()

    else:
        Hsquare = numpy.sum(H.transpose(0, 2, 1) @ H, axis=0)

        HTXT = numpy.sum(H.transpose(0, 2, 1) @ P.transpose(0, 2, 1), axis=0)
        beta = numpy.linalg.solve(numpy.eye(Hsquare.shape[0]) * (1 / C) + Hsquare, HTXT)

    del Hsquare
    del HTXT
    del H

    # endregion

    return beta


def WSI_OMRAE(P, flag_GPU, L, C, activation_function, parameter_of_activation_function):
    number_of_training_data, d1, d2 = numpy.float64(P.shape)  # 读取输入神经元的个数（也就是特征个数）和训练数据样本个数
    sw = calculate_sw(P)

    # region Randomly generate the input weight and bias.
    input_weight = numpy.random.rand(int(L), int(d1)) * 2 - 1
    bias = numpy.random.rand(int(L), int(d2)) * 2 - 1
    # endregion

    # region Compute hidden-layer output
    tempH = input_weight @ P + bias
    del input_weight
    del bias

    if activation_function.lower() == 'sigmoid':
        H = 1 / (1 + numpy.exp(-parameter_of_activation_function * tempH))
    elif activation_function.lower() == 'scaling sigmoid':
        scaling_parameter = parameter_of_activation_function / numpy.abs(tempH).max()
        H = 1 / (1 + numpy.exp(-scaling_parameter * tempH))
    elif activation_function.lower() == 'sine':
        H = numpy.sin(tempH)
    elif activation_function.lower() == 'linear':
        H = tempH
    else:
        raise SystemError('Unknown activation function.')

    del tempH
    # endregion

    # region Derive the output weight.
    H = torch.from_numpy(H)
    P = torch.from_numpy(P)

    beta = torch.rand([int(L), int(d1)], dtype=torch.float64) * 2 - 1
    # beta.requires_grad_(True)
    size_of_batches = 100
    learning_rate = 1e-2
    beta_dw = 0
    number_of_batches = int(number_of_training_data // size_of_batches)
    if number_of_batches == 0:
        number_of_batches = 1
    sw = torch.from_numpy(sw)
    epochs = 100

    if flag_GPU == 1:
        H = H.to('cuda')
        P = P.to('cuda')
        sw = sw.to('cuda')
        beta = beta.to('cuda')

    for k_number_of_epochs in range(int(epochs)):

        epoch_time_start = time.time()

        batch_index = numpy.random.permutation(int(number_of_training_data))

        for k_number_of_batches in range(number_of_batches):
            if k_number_of_batches == number_of_batches:
                H_batch = H[batch_index[k_number_of_batches * size_of_batches:], :]
                P_batch = P[batch_index[k_number_of_batches * size_of_batches:], :]

            else:
                H_batch = H[batch_index[
                            k_number_of_batches * size_of_batches: (k_number_of_batches + 1) * size_of_batches], :]
                P_batch = P[batch_index[
                            k_number_of_batches * size_of_batches: (k_number_of_batches + 1) * size_of_batches], :]

            beta_grad = C * torch.sum(
                H_batch @ H_batch.permute((0, 2, 1)) @ beta - H_batch @ P_batch.permute((0, 2, 1)), dim=0) + beta @ sw

            beta_dw = 0.9 * beta_dw + beta_grad / beta_grad.abs().max()
            beta = beta - learning_rate * beta_dw

        if k_number_of_epochs % 10 == 0:
            learning_rate = learning_rate / 10

        epoch_time = time.time() - epoch_time_start
        print('epoch time:', epoch_time)

    beta = beta.cpu()
    beta = beta.detach().numpy()
    del beta_dw
    del P
    del H
    del H_batch
    del P_batch
    del beta_grad

    return beta


def WID_OMRAE(P, flag_GPU, L, C, activation_function, parameter_of_activation_function):
    number_of_training_data, d1, d2 = numpy.float64(P.shape)

    # region Randomly generate the input weight and bias.
    input_weight = numpy.random.rand(int(L), int(d1)) * 2 - 1
    bias = numpy.random.rand(int(L), int(d2)) * 2 - 1
    # endregion

    # region Compute hidden-layer output
    tempH = input_weight @ P + bias
    del input_weight
    del bias

    if activation_function.lower() == 'sigmoid':
        H = 1 / (1 + numpy.exp(-parameter_of_activation_function * tempH))
    elif activation_function.lower() == 'scaling sigmoid':
        scaling_parameter = parameter_of_activation_function / numpy.abs(tempH).max()
        H = 1 / (1 + numpy.exp(-scaling_parameter * tempH))
    elif activation_function.lower() == 'sine':
        H = numpy.sin(tempH)
    elif activation_function.lower() == 'linear':
        H = tempH
    else:
        raise SystemExit('Unknown activation function.')

    del tempH
    # endregion

    # region Derive the output weight.
    H = torch.from_numpy(H)
    P = torch.from_numpy(P)

    beta = torch.rand([int(L), int(d1)], dtype=torch.float64) * 2 - 1
    size_of_batches = 100
    number_of_batches = int(number_of_training_data // size_of_batches)
    if number_of_batches == 0:
        number_of_batches = 1
    epochs = 100

    if flag_GPU == 1:
        H = H.to('cuda')
        P = P.to('cuda')
        beta = beta.to('cuda')

    beta.requires_grad_(True)

    # Using Adam for speed up
    optimizer = torch.optim.Adam([{'params': beta, 'lr': 0.005}],
                                 weight_decay=0.0005, )
    scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.2, last_epoch=-1)

    for k_number_of_epochs in range(int(epochs)):

        epoch_time_start = time.time()

        batch_index = numpy.random.permutation(int(number_of_training_data))

        for k_number_of_batches in range(number_of_batches):

            if k_number_of_batches == number_of_batches:
                H_batch = H[batch_index[k_number_of_batches * size_of_batches:], :]
                P_batch = P[batch_index[k_number_of_batches * size_of_batches:], :]

            else:
                H_batch = H[batch_index[
                            k_number_of_batches * size_of_batches: (k_number_of_batches + 1) * size_of_batches - 1], :]
                P_batch = P[batch_index[
                            k_number_of_batches * size_of_batches: (k_number_of_batches + 1) * size_of_batches - 1], :]

            W, D = calculate_W_D(P_batch)

            ###########################################################################
            beta_P = beta @ P_batch
            beta_P_T = beta_P.permute(0, 2, 1).contiguous()

            beta_P_num, beta_P_d1, beta_P_d2 = beta_P.shape
            _beta_P_T_num, beta_P_T_d1, beta_P_T_d2 = beta_P_T.shape

            beta_P = beta_P.view((beta_P_num, beta_P_d1 * beta_P_d2))
            beta_P_T = beta_P_T.view((_beta_P_T_num, beta_P_T_d1 * beta_P_T_d2))

            loss = 0.5 * C * torch.sum(torch.norm(beta.T @ H_batch - P_batch, p='fro')) + 0.5 * (
                        beta_P.T @ (D - W) @ beta_P_T).trace()
            ############################################################################
            # beta.retain_grad()
            optimizer.zero_grad()
            loss.backward(loss.clone().detach())
            optimizer.step()
        scheduler1.step()

        epoch_time = time.time() - epoch_time_start
        print('a epoch need time:', epoch_time)

    beta = beta.cpu()
    beta = beta.detach().numpy()
    del P
    del H
    del H_batch
    del P_batch

    return beta
