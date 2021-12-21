import numpy
import numpy as np
import itertools
import random


######## These constraints define the output specification for the coarse verification problem
import scipy.cluster.vq


# ########################### this is for cifar-10
# save_list = np.load("/data/cifar10_coarse_Rob_over_all_save_list.npy", allow_pickle=True)
# save_list = save_list[:,0:4]
# temp = save_list[:,1]
# for index in range(len(temp)):
#     temp[index] = temp[index][0]
# save_list[:,1] = temp

# ########################### this is for fmnist
save_list = np.load("/data/fMNIST_coarse_Rob_over_all_save_list.npy", allow_pickle=True)

#save_list = list(save_list)
#save_list = np.load("examples/Coarse_Rob_fMNIST_tarAtt_cw/fMNIST_coarse_Rob_over_all_save_list_withLinf.npy", allow_pickle=True)

####### define here a dictionary and define the values as lists
dict_data = {}
for idx in range(len(save_list)):
    item = save_list[idx]
    #item[1][0] = int(item[1][0])
    #item = list(item)
    #dict_data[(int(item[1][0]), item[2])].append(item[3])
    dict_data[((item[1]), item[2])]=[]

####### append to the list
for idx in range(len(save_list)):
    item = save_list[idx]
    #item[1][0] = int(item[1][0])
    #item = list(item)
    #dict_data[(int(item[1][0]), item[2])].append(item[3])
    dict_data[((item[1]), item[2])].append(item[3])

####### take the mean
dict_data_final = {}
for idx in range(len(save_list)):
    item = save_list[idx]
    #item[1][0] = int(item[1][0])
    #item = list(item)
    #dict_data[(int(item[1][0]), item[2])].append(item[3])
    dict_data_final[((item[1]), item[2])]=[np.mean(dict_data[(int(item[1]), item[2])]), np.std(dict_data[(int(item[1]), item[2])])]

####### convert to np array
labels_strings = ["Tshirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "ankle-boot", "bag", "Sneaker"]
labels_strings_np = np.array(labels_strings)

final_table_mean = np.zeros(shape=(10,10))
final_table_sttd = np.zeros(shape=(10,10))
for i in range(10):
    for j in range(10):
        if i != j:
            final_table_mean[i ,j] = dict_data_final[i,j][0]
            final_table_sttd[i, j] = dict_data_final[i,j][1]
# below will replace the zeros with nan so as not to avergaed over
final_table_mean[np.where(final_table_mean == 0)] = numpy.nan
final_table_sttd[np.where(final_table_sttd == 0)] = numpy.nan
# keep only four digits
final_table_mean_L2 = final_table_mean.round(decimals=3)
R_2 = final_table_mean_L2
final_table_sttd_L2 = final_table_sttd.round(decimals=3)
Delta_2 = final_table_sttd_L2*final_table_sttd_L2
# # convert to list to add labels:
# final_table_mean_list = list(final_table_mean)
# final_table_mean_list.insert(0,[0,labels_strings])


### let the list of M_c = 3 lists represents mapping_T_test



# mapping_T_test_old = [[0,1,2,3,4,6],[5,7,9],[8]]
# mapping_T_test = [[0,1,3],[5,7,9],[2,4,6,8]]

# build a function: input: mapping_T, R_2, Delta_2, M_c ; output:C_R_2, C_Delta_2 PI_2, sum(PI_2), flag if an entry is below 0
def C_PI_calculator(R, Delta, mapping_T, M_c):
    """

    :param R: np array with nan diagonal so as to count for the zeros
    :param Delta: np array with nan diagonal so as to count for the zeros
    :param mapping_T: list of M_c lists
    :param M_c: number of super labels
    :return: C_R_, C_D_, PI, and sum of PI, flag
    """
    # 1 C_R_2 and C_D_2 from mapping_T_test
    C_R_ = np.zeros(shape=(M_c, M_c))
    C_D_ = np.zeros(shape=(M_c, M_c))
    for i in range(M_c):
        for j in range(M_c):
            sup_ind_i = mapping_T[i]
            sup_ind_j = mapping_T[j]
            C_R_[i, j] = np.nanmean(R[sup_ind_i][:, sup_ind_j])
            C_D_[i, j] = np.nanmean(Delta[sup_ind_i][:, sup_ind_j])

    # ############## replace nan with zeros in matrix C
    # for pair in np.argwhere(np.isnan(C_R_)):
    #     C_R_[pair[0],pair[1]] = 0
    # for pair in np.argwhere(np.isnan(C_D_)):
    #     C_D_[pair[0],pair[1]] = 0

    # 2 matrix PI
    PI_ = np.zeros(shape=(M_c, M_c))
    for i in range(0, M_c):
        for j in range(0, M_c):
            if i != j:
                PI_[i, j] = (C_R_[i, j] - C_R_[i, i]) \
                            / (C_D_[i, j] + C_D_[i, i])

    # 3 PI_sum_metric
    PI_sum_metric = np.nansum(PI_)

    # flag for less than zeros
    flag_of_less_than_zero_element = 0
    if np.any(PI_< 0.0):
        flag_of_less_than_zero_element = 1

    return C_R_, C_D_, PI_, PI_sum_metric, flag_of_less_than_zero_element
# TT = [[3,5], [0,2,8,9], [2,4,6,7]]
# C_PI_calculator(R_2, Delta_2, TT, 3)
print('break')

#################################################################################################################################
# ############################# brute force
#################################################################################################################################

def sorted_k_partitions(seq, k):
    """Returns a list of all unique k-partitions of `seq`.

    Each partition is a list of parts, and each part is a tuple.

    The parts in each individual partition will be sorted in shortlex
    order (i.e., by length first, then lexicographically).

    The overall list of partitions will then be sorted by the length
    of their first part, the length of their second part, ...,
    the length of their last part, and then lexicographically.
    """
    n = len(seq)
    groups = []  # a list of lists, currently empty

    def generate_partitions(i):
        if i >= n:
            yield list(map(tuple, groups))
        else:
            if n - i > k - len(groups):
                for group in groups:
                    group.append(seq[i])
                    yield from generate_partitions(i + 1)
                    group.pop()

            if len(groups) < k:
                groups.append([seq[i]])
                yield from generate_partitions(i + 1)
                groups.pop()

    result = generate_partitions(0)

    # Sort the parts in each partition in shortlex order
    result = [sorted(ps, key = lambda p: (len(p), p)) for ps in result]
    # Sort partitions by the length of each part, then lexicographically.
    result = sorted(result, key = lambda ps: (*map(len, ps), ps))

    return result





print('break')


##################################################################################################
################################3 Greedy algorithm that considers all comibations of size Mc=3 and M=10#####
##################################################################################################
######### loop over all non-empty cases possibilities (should be ~9000, see ppt slides)
M_c = 4
possibilities = sorted_k_partitions(list(range(0,10)),M_c)

list_of_all_possibilities = []
for possibility in possibilities:
    list_of_all_possibilities.append([list(ele) for ele in possibility])


max_PI = 0
T_current_best = []
idx = 0
for combo in list_of_all_possibilities:
    idx = idx + 1

    #########B calculate C_R, C_D, PI_1, PI_sum_metric,  flag_of_less_than_zero_element
    _, _, PI_, PI_sum_metric, flag_of_less_than_zero_element = C_PI_calculator(
        R_2, Delta_2, combo, M_c)

    #if flag_of_less_than_zero_element == 0:

    if PI_sum_metric > max_PI and flag_of_less_than_zero_element == 0:
        max_PI = PI_sum_metric
        T_current_best = combo

        print("index", idx, "Mapping = ", combo, "PI_sum_metric = ", [PI_sum_metric], "; validity = ", flag_of_less_than_zero_element, '; current max is = ', max_PI, "; at mapping: ", T_current_best)



print("break")




