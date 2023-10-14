from sklearn import metrics
import scipy.io
import numpy
import torch

class Dataset():

    def __init__(self, P, T, TV_P, TV_T):
        self.P = P
        self.T = T
        self.TV_P = TV_P
        self.TV_T = TV_T

class my_Struct():
    pass

def set_random_seed(seed):
    """
    虽然我设置了同样的种子，但是我发现在不同的电脑上，得到的结果依然不一致
    虽然不同电脑，同一个随机种子，使用torch.rand(100)得到的结果是一样的。╮(╯▽╰)╭
    """
    if seed is not None:
        # ---- set random seed
        # random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def my_roc_curve(deci,label_y, label_of_positive_samples, type_of_deci):
    # deci is the actual output
    # label_y is the desired output, i.e., the true label. It
    # label_of_positive_samples is the label of the positive_samples
    # type_of_deci is the actual output type, including
    #   1: the probability belonging to positive sample.
    #   2: the distance. The smaller distance, the closer to positive sample.
    if type_of_deci == 1:
        pass # 属于正样本的概率值从大到小排序
    elif type_of_deci == 2:
        deci = 1/deci
    else:
        raise SystemExit('Unknown switch case.')

    fpr, tpr, thresholds = metrics.roc_curve(label_y.flatten(), deci.flatten(), pos_label=int(label_of_positive_samples))
    auc = metrics.auc(fpr, tpr)
    return auc, fpr, tpr


def my_loadmat_for_grid_search_results(mat_path):
    my_results = scipy.io.loadmat(mat_path, mat_dtype=True, struct_as_record=False, squeeze_me=True)
    #  #struct_as_record=False, squeeze_me=True #simplify_cells=True #matlab_compatible=False
    # 这是我测试出来我觉着最好的结果
    # 读进来的my_results包含"AE_parameters","ML_OCELM_parameters","testing_accuracy","ave_testing_accuracy"
    # AE_parameters是一个ndarray(number_of_AEs,)数组，每一个元素是一个my_Struct结构体，包含以下变量：
        # flag_GPU，int型
        # activation_function，str型
        # parameter_of_activation_function，numpy.float64型
        # number_of_hidden_layer_neurons，numpy.float64型
        # C，numpy.float64型

    # ML_OCELM_parameters是一个my_Struct结构体，包含以下变量：
        # activation_function，是一个ndarray(number_of_AEs,)数组，每一个元素是str型
        # parameter_of_activation_function，是一个ndarray(number_of_AEs,)数组，每一个元素是numpy.float64型
        # flag_GPU，int型
        # C，numpy.float64型
        # mu，numpy.float64型

    # testing_accuracy是一个ndarray(number_of_repeats,)数组，每一个元素是一个my_Struct结构体，包含以下变量：
        # F1，numpy.float64型
        # Gmean，numpy.float64型
        # P，numpy.float64型
        # R，numpy.float64型
        # auc，numpy.float64型
        # overall_accuracy，numpy.float64型

    # ave_testing_accuracy是一个my_Struct结构体，包含以下变量：
        # overall_accuracy，numpy.float64型
        # Gmean，numpy.float64型
        # R，numpy.float64型
        # P，numpy.float64型
        # F1，numpy.float64型
        # auc，numpy.float64型

    #region AE_parameters
    AE_parameters = numpy.array([])
    if type(my_results['AE_parameters']) is numpy.ndarray: #这说明有多个AE
        for k_AE_parameters in my_results['AE_parameters']:
            AE_parameters = numpy.append(AE_parameters, my_Struct())
            AE_parameters[-1].flag_GPU = int(k_AE_parameters.flag_GPU)
            AE_parameters[-1].activation_function = str(k_AE_parameters.activation_function)
            AE_parameters[-1].parameter_of_activation_function = numpy.float64(k_AE_parameters.parameter_of_activation_function)
            AE_parameters[-1].number_of_hidden_layer_neurons = numpy.float64(k_AE_parameters.number_of_hidden_layer_neurons)
            AE_parameters[-1].C = numpy.float64(k_AE_parameters.C)
    else:
        AE_parameters = numpy.append(AE_parameters, my_Struct())
        AE_parameters[-1].flag_GPU = int(my_results['AE_parameters'].flag_GPU)
        AE_parameters[-1].activation_function = str(my_results['AE_parameters'].activation_function)
        AE_parameters[-1].parameter_of_activation_function = numpy.float64(my_results['AE_parameters'].parameter_of_activation_function)
        AE_parameters[-1].number_of_hidden_layer_neurons = numpy.float64(my_results['AE_parameters'].number_of_hidden_layer_neurons)
        AE_parameters[-1].C = numpy.float64(my_results['AE_parameters'].C)
    #endregion

    # region ML_OCELM_parameters
    ML_OCELM_parameters = my_Struct()
    ML_OCELM_parameters.activation_function = numpy.array([])
    ML_OCELM_parameters.parameter_of_activation_function = numpy.array([])
    ML_OCELM_parameters.flag_GPU = int(my_results['ML_OCELM_parameters'].flag_GPU)
    ML_OCELM_parameters.C = numpy.float64(my_results['ML_OCELM_parameters'].C)
    ML_OCELM_parameters.mu = numpy.float64(my_results['ML_OCELM_parameters'].mu)

    if type(my_results['ML_OCELM_parameters'].activation_function) is numpy.ndarray:  # 这说明有多个AE
        for k_activation_function, k_parameter_of_activation_function in \
                zip(my_results['ML_OCELM_parameters'].activation_function, \
                    my_results['ML_OCELM_parameters'].parameter_of_activation_function):

            ML_OCELM_parameters.activation_function = numpy.append(ML_OCELM_parameters.activation_function, None)
            ML_OCELM_parameters.parameter_of_activation_function = numpy.append(ML_OCELM_parameters.parameter_of_activation_function, None)

            ML_OCELM_parameters.activation_function[-1] = str(k_activation_function)
            ML_OCELM_parameters.parameter_of_activation_function[-1] = numpy.float64(k_parameter_of_activation_function)

    else:
        ML_OCELM_parameters.activation_function = numpy.append(ML_OCELM_parameters.activation_function, None)
        ML_OCELM_parameters.parameter_of_activation_function = numpy.append(ML_OCELM_parameters.parameter_of_activation_function, None)

        ML_OCELM_parameters.activation_function[-1] = str(my_results['ML_OCELM_parameters'].activation_function)
        ML_OCELM_parameters.parameter_of_activation_function[-1] = numpy.float64(my_results['ML_OCELM_parameters'].parameter_of_activation_function)
    #endregion

    #region testing_accuracy
    testing_accuracy = numpy.array([])
    for k_testing_accuracy in my_results['testing_accuracy']:
        testing_accuracy = numpy.append(testing_accuracy, my_Struct())

        testing_accuracy[-1].overall_accuracy = numpy.float64(k_testing_accuracy.overall_accuracy)
        testing_accuracy[-1].Gmean = numpy.float64(k_testing_accuracy.Gmean)
        testing_accuracy[-1].R = numpy.float64(k_testing_accuracy.R)
        testing_accuracy[-1].P = numpy.float64(k_testing_accuracy.P)
        testing_accuracy[-1].F1 = numpy.float64(k_testing_accuracy.F1)
        testing_accuracy[-1].auc = numpy.float64(k_testing_accuracy.auc)
    #endregion

    # region ave_testing_accuracy
    ave_testing_accuracy = my_Struct()
    ave_testing_accuracy.overall_accuracy = numpy.float64(my_results['ave_testing_accuracy'].overall_accuracy)
    ave_testing_accuracy.Gmean = numpy.float64(my_results['ave_testing_accuracy'].Gmean)
    ave_testing_accuracy.R = numpy.float64(my_results['ave_testing_accuracy'].R)
    ave_testing_accuracy.P = numpy.float64(my_results['ave_testing_accuracy'].P)
    ave_testing_accuracy.F1 = numpy.float64(my_results['ave_testing_accuracy'].F1)
    ave_testing_accuracy.auc = numpy.float64(my_results['ave_testing_accuracy'].auc)
    # endregion

    return AE_parameters, ML_OCELM_parameters, testing_accuracy, ave_testing_accuracy


def my_loadmat_for_optimal_results(mat_path):
    my_results = scipy.io.loadmat(mat_path, mat_dtype=True, struct_as_record=False, squeeze_me=True)
    #  #struct_as_record=False, squeeze_me=True #simplify_cells=True #matlab_compatible=False
    # 这是我测试出来我觉着最好的结果
    # 读进来的my_results包含"optimal_AE_parameters","optimal_ML_OCELM_parameters","optimal_number_of_AEs","optimal_testing_accuracy","optimal_ave_testing_accuracy"
    # optimal_AE_parameters是一个ndarray(number_of_AEs,)数组，每一个元素是一个my_Struct结构体，包含以下变量：
        # flag_GPU，int型
        # activation_function，str型
        # parameter_of_activation_function，numpy.float64型
        # number_of_hidden_layer_neurons，numpy.float64型
        # C，numpy.float64型

    # optimal_ML_OCELM_parameters是一个my_Struct结构体，包含以下变量：
        # activation_function，是一个ndarray(number_of_AEs,)数组，每一个元素是str型
        # parameter_of_activation_function，是一个ndarray(number_of_AEs,)数组，每一个元素是numpy.float64型
        # flag_GPU，int型
        # C，numpy.float64型
        # mu，numpy.float64型

    # optimal_number_of_AEs，int型

    # optimal_testing_accuracy是一个ndarray(number_of_repeats,)数组，每一个元素是一个my_Struct结构体，包含以下变量：
        # F1，numpy.float64型
        # Gmean，numpy.float64型
        # P，numpy.float64型
        # R，numpy.float64型
        # auc，numpy.float64型
        # overall_accuracy，numpy.float64型

    # optimal_ave_testing_accuracy是一个my_Struct结构体，包含以下变量：
        # overall_accuracy，numpy.float64型
        # Gmean，numpy.float64型
        # R，numpy.float64型
        # P，numpy.float64型
        # F1，numpy.float64型
        # auc，numpy.float64型

    #region optimal_AE_parameters
    optimal_AE_parameters = numpy.array([])
    if type(my_results['optimal_AE_parameters']) is numpy.ndarray: #这说明有多个AE
        for k_AE_parameters in my_results['optimal_AE_parameters']:
            optimal_AE_parameters = numpy.append(optimal_AE_parameters, my_Struct())
            optimal_AE_parameters[-1].flag_GPU = int(k_AE_parameters.flag_GPU)
            optimal_AE_parameters[-1].activation_function = str(k_AE_parameters.activation_function)
            optimal_AE_parameters[-1].parameter_of_activation_function = numpy.float64(k_AE_parameters.parameter_of_activation_function)
            optimal_AE_parameters[-1].number_of_hidden_layer_neurons = numpy.float64(k_AE_parameters.number_of_hidden_layer_neurons)
            optimal_AE_parameters[-1].C = numpy.float64(k_AE_parameters.C)
    else:
        optimal_AE_parameters = numpy.append(optimal_AE_parameters, my_Struct())
        optimal_AE_parameters[-1].flag_GPU = int(my_results['optimal_AE_parameters'].flag_GPU)
        optimal_AE_parameters[-1].activation_function = str(my_results['optimal_AE_parameters'].activation_function)
        optimal_AE_parameters[-1].parameter_of_activation_function = numpy.float64(my_results['optimal_AE_parameters'].parameter_of_activation_function)
        optimal_AE_parameters[-1].number_of_hidden_layer_neurons = numpy.float64(my_results['optimal_AE_parameters'].number_of_hidden_layer_neurons)
        optimal_AE_parameters[-1].C = numpy.float64(my_results['optimal_AE_parameters'].C)
    #endregion

    # region optimal_ML_OCELM_parameters
    optimal_ML_OCELM_parameters = my_Struct()
    optimal_ML_OCELM_parameters.activation_function = numpy.array([])
    optimal_ML_OCELM_parameters.parameter_of_activation_function = numpy.array([])
    optimal_ML_OCELM_parameters.flag_GPU = int(my_results['optimal_ML_OCELM_parameters'].flag_GPU)
    optimal_ML_OCELM_parameters.C = numpy.float64(my_results['optimal_ML_OCELM_parameters'].C)
    optimal_ML_OCELM_parameters.mu = numpy.float64(my_results['optimal_ML_OCELM_parameters'].mu)

    if type(my_results['optimal_ML_OCELM_parameters'].activation_function) is numpy.ndarray:  # 这说明有多个AE
        for k_activation_function, k_parameter_of_activation_function in \
                zip(my_results['optimal_ML_OCELM_parameters'].activation_function, \
                    my_results['optimal_ML_OCELM_parameters'].parameter_of_activation_function):

            optimal_ML_OCELM_parameters.activation_function = numpy.append(optimal_ML_OCELM_parameters.activation_function, None)
            optimal_ML_OCELM_parameters.parameter_of_activation_function = numpy.append(optimal_ML_OCELM_parameters.parameter_of_activation_function, None)

            optimal_ML_OCELM_parameters.activation_function[-1] = str(k_activation_function)
            optimal_ML_OCELM_parameters.parameter_of_activation_function[-1] = numpy.float64(k_parameter_of_activation_function)

    else:
        optimal_ML_OCELM_parameters.activation_function = numpy.append(optimal_ML_OCELM_parameters.activation_function, None)
        optimal_ML_OCELM_parameters.parameter_of_activation_function = numpy.append(optimal_ML_OCELM_parameters.parameter_of_activation_function, None)

        optimal_ML_OCELM_parameters.activation_function[-1] = str(my_results['optimal_ML_OCELM_parameters'].activation_function)
        optimal_ML_OCELM_parameters.parameter_of_activation_function[-1] = numpy.float64(my_results['optimal_ML_OCELM_parameters'].parameter_of_activation_function)
    #endregion

    #region optimal_number_of_AEs
    optimal_number_of_AEs = int(my_results['optimal_number_of_AEs'])
    #endregion

    #region optimal_testing_accuracy
    optimal_testing_accuracy = numpy.array([])
    for k_testing_accuracy in my_results['optimal_testing_accuracy']:
        optimal_testing_accuracy = numpy.append(optimal_testing_accuracy, my_Struct())

        optimal_testing_accuracy[-1].overall_accuracy = numpy.float64(k_testing_accuracy.overall_accuracy)
        optimal_testing_accuracy[-1].Gmean = numpy.float64(k_testing_accuracy.Gmean)
        optimal_testing_accuracy[-1].R = numpy.float64(k_testing_accuracy.R)
        optimal_testing_accuracy[-1].P = numpy.float64(k_testing_accuracy.P)
        optimal_testing_accuracy[-1].F1 = numpy.float64(k_testing_accuracy.F1)
        optimal_testing_accuracy[-1].auc = numpy.float64(k_testing_accuracy.auc)
    #endregion

    # region ave_testing_accuracy
    optimal_ave_testing_accuracy = my_Struct()
    optimal_ave_testing_accuracy.overall_accuracy = numpy.float64(my_results['optimal_ave_testing_accuracy'].overall_accuracy)
    optimal_ave_testing_accuracy.Gmean = numpy.float64(my_results['optimal_ave_testing_accuracy'].Gmean)
    optimal_ave_testing_accuracy.R = numpy.float64(my_results['optimal_ave_testing_accuracy'].R)
    optimal_ave_testing_accuracy.P = numpy.float64(my_results['optimal_ave_testing_accuracy'].P)
    optimal_ave_testing_accuracy.F1 = numpy.float64(my_results['optimal_ave_testing_accuracy'].F1)
    optimal_ave_testing_accuracy.auc = numpy.float64(my_results['optimal_ave_testing_accuracy'].auc)
    # endregion

    return optimal_AE_parameters, optimal_ML_OCELM_parameters, optimal_number_of_AEs, optimal_testing_accuracy, optimal_ave_testing_accuracy


def calculate_sw(P):
    number_of_training_data, d1, d2 = numpy.float64(P.shape)

    mean = numpy.mean(P, axis=0)

    sw = P - mean
    sw = sw @ sw.transpose(0, 2, 1)
    sw = numpy.sum(sw, axis=0)

    return sw

def calculate_W_D(P):
    number_of_training_data, d1, d2 = P.shape
    # P = P.reshape(number_of_training_data, -1)

    W = torch.ones((number_of_training_data, number_of_training_data), dtype=torch.float64, device=P.device)
    temp = torch.sum(W, dim=1)
    D = torch.diag(temp)

    return W,D
