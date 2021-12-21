import numpy
import numpy as np
import itertools
import random


######## These constraints define the output specification for the coarse verification problem
import scipy.cluster.vq


########################### this is for cifar-10
save_list = np.load("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/cifar10_coarse_Rob_over_all_save_list.npy", allow_pickle=True)
save_list = save_list[:,0:4]
temp = save_list[:,1]
for index in range(len(temp)):
    temp[index] = temp[index][0]
save_list[:,1] = temp

# ########################### this is for fmnist
#save_list = np.load("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/fMNIST_coarse_Rob_over_all_save_list.npy", allow_pickle=True)

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
TTT = [[8],[5,7,9],[0,1,2,3,4,6]]
C_PI_calculator(R_2, Delta_2, TTT, 3)
print('break')


############################################################################################################################################################################################################
############################################################################################################################################################################################################
############################################################################################################################################################################################################
#################################################################### mappings to save for showing the heatmaps: ####################################################################
############################################################################################################################################################################################################
############################################################################################################################################################################################################
############################################################################################################################################################################################################

# ################# fmnist #################
# _, _, PI_fmnist_Mc3_best, _, _ = C_PI_calculator(
#     R_2, Delta_2, [[1,3],[5,7,9],[0,2,4,6,8]], M_c=3)
#
# _, _, PI_fmnist_Mc4_best, _, _ = C_PI_calculator(
#     R_2, Delta_2, [[4,8],[1,3],[5,7,9],[0,2,6]], M_c=4)
#
# np.save("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/Pi_best_fmnist_matricies.npy",[PI_fmnist_Mc3_best,PI_fmnist_Mc4_best])
#

# ################# cifar-10 #################
# _, _, PI_cifar10_Mc3_best, _, _ = C_PI_calculator(
#     R_2, Delta_2, [[2,3,4,5,6,7], [0,8], [1,9]], M_c=3)
#
# _, _, PI_cifar10_Mc4_best, _, _ = C_PI_calculator(
#     R_2, Delta_2, [[1,9], [3,5], [4,7],[0,2,6,8]], M_c=4)

# _, _, PI_cifar10_Mc3_best, _, _ = C_PI_calculator(
#     R_2, Delta_2, [[6, 9, 8, 4, 0, 2, 7], [3, 1], [5]], M_c=3)
#
# _, _, PI_cifar10_Mc4_best, _, _ = C_PI_calculator(
#     R_2, Delta_2, [[5], [8, 3, 0, 9, 7, 1], [6, 4, 2]], M_c=4)

#np.save("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/Pi_best_cifar10_matricies.npy",[PI_cifar10_Mc3_best,PI_cifar10_Mc4_best])




# ############################################################################################################################################################################################################
# ############################################################################################################################################################################################################
# ############################################################################################################################################################################################################
# #################################################################### Algorithm for best groupings: ####################################################################
# ############################################################################################################################################################################################################
# ############################################################################################################################################################################################################
# ############################################################################################################################################################################################################
# # #### from brute force, best mapping with cifar-10 Mc = 3 is with sum of PI = 14.8439:
# # T_star = [[0,8], [1, 9], [2,3,4,5,6,7]]
# # C_R_star, C_D_star, PI_star, PI_sum_metric_star, flag_of_less_than_zero_element_star = C_PI_calculator(
# #     R_2, Delta_2, T_star, 3)
# # #### from brute force, best mapping with Mc = 3 is with sum of PI = 13.4068:
# # T_star = [[7, 5, 9], [1, 3], [4, 8, 6, 0, 2]]
# # C_R_star, C_D_star, PI_star, PI_sum_metric_star, flag_of_less_than_zero_element_star = C_PI_calculator(
# #     R_2, Delta_2, T_star, 3)
#
# # #### from brute force, best mapping with Mc = 4 is with sum of PI = 24.48525:
# # T_star = [[0, 6, 8], [1, 3], [2, 4], [5, 7, 9]]
# # C_R_star, C_D_star, PI_star, PI_sum_metric_star, flag_of_less_than_zero_element_star = C_PI_calculator(
# #     R_2, Delta_2, T_star, 4)
#
#
#
# ### T_init to test
#
# #T_init = [[3,4,6],[0,1,9],[2,5,8, 7]]
# #T_init = [[0,1,9],[3,4,6],[2,5,8, 7]] # this is currently reaches a sub-optimal solution
# #T_init = [[2,3,4,5,6,7,8,9],[0],[1]]# this should
# #T_init = [[0,1,2,3,4], [5,6], [8,7,9]] # THIS WORKS
# #T_init = [[0], [5,6,1,2,3,4], [8,7,9]] # THIS WORKS
# #T_init = [[0], [4], [8,7,9,5,6,1,2,3]] # this is only sub-optimal
# #T_init = [[0, 3, 1], [4, 6, 2, 5, 7], [8, 9]]
# #T_init = [[0],[1],[2,3,4,5,6,7,8,9]]
# #T_init = [[0, 2, 3, 4, 6, 8], [1], [5, 7, 9]]
#
# ## MC = 3 init
# #T_init = [[0,1,9],[3,4,6],[2,5,8, 7]] # this is currently reaches a sub-optimal solution
# #T_init = [[2,3,4,5,6,7,8,9],[0],[1]]# this should
# #T_init = [[0,1,2,3,4], [5,6], [8,7,9]] # THIS WORKS
# #T_init = [[0], [5,6,1,2,3,4], [8,7,9]] # THIS WORKS
#
# ## MC = 4 init
# #T_init = [[0], [4], [8], [7,9,5,6,1,2,3]]
# #T_init = [[0,1,2,3], [4], [5,6], [7,8,9]]
# #T_init = [[0,1,2,3,4,5,6], [8], [9], [7]]
#
#
# # # this is for M_c = 2
# # ranom_list = random.sample(range(10), 10)
# # T_init = [ranom_list[0:5], ranom_list[5:10]]
#
#
# # # this is for M_c = 3
# # ranom_list = random.sample(range(10), 10)
# # T_init = [ranom_list[0:3], ranom_list[3:5], ranom_list[5:10]]
#
# # # this is for M_c = 4
# # ranom_list = random.sample(range(10), 10)
# # T_init = [ranom_list[0:3], ranom_list[3:5], ranom_list[5:7], ranom_list[7:10]]
#
#
# # this is for M_c = 5
# ranom_list = random.sample(range(10), 10)
# T_init = [ranom_list[0:3], ranom_list[3:5], ranom_list[5:7], ranom_list[7:8],ranom_list[8:10]]
#
#
# # FLAG If the initial mapping_T has empty set
# for ls in T_init:
#     if ls == []:
#         print("WE HAVE THE PROBLEM OF AN EMPTY GROUP,  FIX", T_init)
#
# ###################### If the maximum is reapeated, exit with best mapping
# number_or_iterations = 4500
# mappings_list_to_save = []
# sum_ofPI_toSave       = []
# bad_mappings_list_to_save = []
# bad_sum_ofPI_toSave = []
# ######A start with T = T_spec_clustering with flag = 0
# T_current = T_init
#
# M_c = 5
#
# mappings_list_to_save.append(T_current)
# sum_ofPI_toSave.append(0.0)
# best_mapping_based_onSumOfPI = 0
# PI_sum_metric_current = 0
# for iter in range(0,number_or_iterations):
#
#
#
#     ######### step(1) calculate C_R, C_D, PI_1, PI_sum_metric,  flag_of_less_than_zero_element
#     C_R_current, C_D_current, PI_current, PI_sum_metric_current, flag_of_less_than_zero_element_current = C_PI_calculator(
#         R_2, Delta_2, T_current, M_c)
#     PI_current[PI_current == 0] = np.nan
#     # if flag is not 0; exit while and select the best T based previous attempts and T_spec_clustering
#     flag_of_less_than_zero_element = "Feasible"
#     if flag_of_less_than_zero_element_current != 0:
#         flag_of_less_than_zero_element = "Non-Feasible"
#         #print("GROUPING", T_current, " is bad since we have PI < 0 ; EXIT")
#         #break
#
#
#         #print("Starting from initial random mapping ", T_init," ; BEST is found at iteration =  ", iter, ' ; with mapping = ', T_current, "with SUM = ", PI_sum_metric_current)
#         #break
#
#
#     # if current_best_mapping_based_onSumOfPI is bigger than PI_sum_metric_current and the it is valid, then update current
#     if best_mapping_based_onSumOfPI < PI_sum_metric_current and flag_of_less_than_zero_element_current == 0:
#         best_mapping_based_onSumOfPI =  PI_sum_metric_current
#
#
#
#     ################################### source and destination second try based on the min PI(i,j)
#     # if Second_argmin_in_PI == 1:
#     #     min_value_in_PI = np.nanmin(PI_current[PI_current != np.nanmin(PI_current)])
#     # else:
#     min_value_in_PI = np.nanmin(PI_current)
#
#     Sm_source_Sn_destination  =  np.argwhere(PI_current == min_value_in_PI)[0]
#     S_mSource = T_current[Sm_source_Sn_destination[0]]
#     S_nDestin = T_current[Sm_source_Sn_destination[1]]
#
#     # get m_source = argmin_{m \in S_source , n\in S_destin} R(m,n)
#     temp = R_2[S_mSource][:, S_nDestin]
#     minimum_enty = np.nanmin(temp)
#
#     # if Second_argmin_in_R == 0:
#     #     minimum_enty = np.nanmin(temp)
#     # else:
#     #     minimum_enty = np.nanmin(temp[temp != np.nanmin(temp)])
#     # find indicies of the min entry in R_2
#     m_source  =  np.argwhere(R_2 == minimum_enty)[0][0]
#     n_destin  =  np.argwhere(R_2 == minimum_enty)[0][1]
#
#     ###########E move label m_source to S_T(n_destination) and update T_current
#     # remove m_source
#     T_current_new = [[ele for ele in sub if ele != m_source] for sub in T_current]
#     # append m_source to S_T_(n_destination))
#     for ls in T_current_new:
#         if n_destin in ls:
#             S_dest = ls
#             ls.append(m_source)
#
#     restart = None
#     ######### if current best_mapping is repeated, try starting from different T_init
#     #if T_current_new in mappings_list_to_save and flag_of_less_than_zero_element_current == 0:
#     if PI_sum_metric_current in sum_ofPI_toSave and flag_of_less_than_zero_element_current == 0:
#         #print("Restarting T_current randomly because of a feasible value is repeated ")
#         restart = "Yes - Feasible is repeated"
#         ranom_list = random.sample(range(10), 10)
#         # # this is for M_c = 2
#         #T_current_new = [ranom_list[0:5], ranom_list[5:10]]
#         # # this is for M_c = 3
#         #T_current_new = [ranom_list[0:3], ranom_list[3:7], ranom_list[7:10]]
#         # # this is for M_c = 4
#         #T_current_new = [ranom_list[0:3], ranom_list[3:5],ranom_list[5:7], ranom_list[7:10]]
#         #this is for M_c = 5
#         T_current_new = [ranom_list[0:3], ranom_list[3:5], ranom_list[5:7], ranom_list[7:8],ranom_list[8:10]]
#
#     ######## if a current bad mapping is repeated, then restart T_current
#     #if T_current_new in bad_mappings_list_to_save and flag_of_less_than_zero_element_current == 1:
#     if PI_sum_metric_current in bad_sum_ofPI_toSave and flag_of_less_than_zero_element_current == 1:
#         #print("Restarting T_current randomly because of a non-feasible value is repeated ")
#         restart = "Yes - Non-Feasible is repeated"
#         ranom_list = random.sample(range(10), 10)
#         # # this is for M_c = 2
#         #T_current_new = [ranom_list[0:5], ranom_list[5:10]]
#         # # this is for M_c = 3
#         #T_current_new = [ranom_list[0:3], ranom_list[3:7], ranom_list[7:10]]
#         # # this is for M_c = 4
#         #T_current_new = [ranom_list[0:3], ranom_list[3:5],ranom_list[5:7], ranom_list[7:10]]
#         # this is for M_c = 5
#         T_current_new = [ranom_list[0:3], ranom_list[3:5], ranom_list[5:7], ranom_list[7:8],ranom_list[8:10]]
#
#     ####### logger:
#     #print("Iteration: ", iter, "current best value = ", [best_mapping_based_onSumOfPI] , "; Current and Next mappings are ", [T_current], " ===> ", [T_current_new] , '; STATUS = ', flag_of_less_than_zero_element, "; sum of PI metric = ", [PI_sum_metric_current], "; moving label ", [m_source], " to ", [S_dest])
#
#     ####### logger:
#     print("Iteration: ", iter, "current best value = ", [best_mapping_based_onSumOfPI] , "; Current mapping is ", [T_current], '; STATUS = ', flag_of_less_than_zero_element, "; sum of PI metric = ", [PI_sum_metric_current], "Restart = ", restart)
#
#
#     # only save if it is good
#     if flag_of_less_than_zero_element_current == 0:
#         mappings_list_to_save.append(T_current_new)
#         sum_ofPI_toSave.append(PI_sum_metric_current)
#
#     # if bad, save in bad
#     if flag_of_less_than_zero_element_current == 1:
#         bad_mappings_list_to_save.append(T_current_new)
#         bad_sum_ofPI_toSave.append(PI_sum_metric_current)
#
#     T_current = T_current_new
#
# print("sub-optimal MAPPIG IS OBATINED AT ", best_mapping_based_onSumOfPI)
#
# print("break")


# ##################################################################################################################################################
# ##################################################################################################################################################
# ##################################################################################################################################################
# ############# finding local best value starting from T_website


T_best_CIFAR_Mc4 = [[7, 4], [5, 3], [0, 8], [9, 6, 2, 1]]
alpha_best_CIFAR_Mc4 = 31.371683888903725

T_best_FMNIST_Mc4 = [[8, 4], [6, 2, 0], [5, 7, 9], [3, 1]]
alpha_best_FMNIST_Mc4 = 23.78714621013235

T_best_FMNIST_Mc3 = [[5, 7, 9], [1, 3], [0, 6, 8, 4, 2]]
alpha_best_FMNIST_Mc3 = 13.406841449250665

T_best_cifar_Mc3 = [[3, 2, 7, 5, 6, 4], [9, 1], [8, 0]]
alpha_best_cifar_Mc3 = 14.84390083569589

T_best_cifar_Mc2 = [[1, 9, 8, 0], [4, 7, 3, 5, 6, 2]]
alpha_best_cifar_Mc2 = 4.1365494645172065

T_best_FMNIST_Mc2 = [[1, 8, 3, 4, 6, 2, 0], [5, 9, 7]]
alpha_best_FMNIST_Mc2 = 4.560872105446857

T_best_FMNIST_Mc5 = [[8, 5], [7, 9], [1, 3], [6, 0], [2, 4]]
alpha_best_FMNIST_Mc5 = 37.9487

T_best_CIFAR_Mc5 = [[8, 0], [5, 3], [7], [1, 9], [2, 4, 6]]
alpha_best_CIFAR_Mc5 = 50.29925325499117

print("BREAK")

def trying_McM_OneLabel_Mapps(TT, current_best):

    TT_website_flat = [item for lst in TT for item in lst]
    M = len(TT_website_flat)
    M_c = len(TT)
    storage = []
    for i in range(M):
        # choose the number to be going around for 19 times
        number_to_going_around = TT_website_flat[i]

        # find the super index and index of the number_to_going_around to find the source list
        position = [(ii, jj.index(number_to_going_around)) for ii, jj in enumerate(TT) if
                    number_to_going_around in jj]
        # we found the source list
        source_list = TT[position[0][0]]

        # loop 19 times for every list in TT_website that is not the source list
        TT_website_to_loop = [lst for lst in TT if lst != source_list]
        for j in range(M_c-1):
            destination_list = TT_website_to_loop[j]

            ### operation of removing from the source list and adding to the destination list
            TT[position[0][0]].remove(number_to_going_around)
            destination_list.append(number_to_going_around)
            # print(np.array(TT_website))

            intermediate_list = []
            for lst in TT:
                tempp = []
                for item in lst:
                    tempp.append(item)
                intermediate_list.append(tempp)

            storage.append(intermediate_list)

            ### operation of putting things back
            TT[position[0][0]].append(number_to_going_around)
            destination_list.remove(number_to_going_around)

    mapping_best = []
    current_max = current_best
    for mapping in storage:
        # if the considered mapping does not have a one-label category, then don't even bother to try
        if 1 not in [len(set_S) for set_S in mapping]:
            _, _, PI_star, PI_sum_metric_star, flag_of_less_than_zero_element_star = C_PI_calculator(R_2, Delta_2, mapping,
                                                                                                     M_c)

            if PI_sum_metric_star > current_max and flag_of_less_than_zero_element_star == 0:
                mapping_best = mapping
                current_max = PI_sum_metric_star
                print('Current best from ', current_best ,' is at ', [PI_sum_metric_star])

    # if the current best mappoing stayed empty, then make it equal to the input of the function
    if mapping_best == []:
        mapping_best = TT
        print("non of the tried combinations in the current batch returned a better mapping than the current")


    return current_max,mapping_best

current_max, mapping_best = trying_McM_OneLabel_Mapps(T_best_CIFAR_Mc4, alpha_best_CIFAR_Mc4)

##### the swapping while loop goes here ==>
################ here goes the while loop; keep looping for a very long time. the stopping criteria: (1) one label group, or (2) after number of tries exceeds some threshold
################ OR the stopping criteria can be if all the 1900 tries are tried and input and output are the same
iteration_counter = 0
current_mapping = mapping_best
current_value   = current_max
while iteration_counter < 1000:
    iteration_counter = iteration_counter + 1
#while best_value > current_value:
    best_value, best_mapping = trying_McM_OneLabel_Mapps(current_mapping, current_value)

    if best_value == current_value:
        print("Further enhancement is not possible. hence, we are exiting")
        break
    if best_value > current_value:
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("ITERATION ", [iteration_counter], '[BEST, CURRENT]', [best_value , current_value])
        print("###################################################################################")

    current_value   = best_value
    current_mapping = best_mapping

print("break")

############################################################################################################################################################################################################
############################################################################################################################################################################################################
############################################################################################################################################################################################################
#######################################################################################################################################
############################################################################################################################################################################################################
############################################################################################################################################################################################################
############################################################################################################################################################################################################




#################################################################################################################################
# ############################# below code only gives the cobinations; its copied to a text file "combos_10_3_string.txt"; Mc = 3
#################################################################################################################################
# #!/usr/bin/python3
# import string
# import copy
# #
# # This generates all the possible placements of
# # balls into boxes (configurations with empty boxes are allowed).
# #
# class BinPartitions:
#
#     def __init__(self, balls, num_bins):
#         self.balls = balls
#         self.bins = [{} for x in range(num_bins)]
#
#     def print_bins(self, bins):
#         L = []
#         for b in bins:
#             buf = ''.join(sorted(b.keys()))
#             L += [buf]
#         print(",".join(L))
#
#     def _gen_helper(self,balls,bins):
#         if len(balls) == 0:
#             self.print_bins(bins)
#         else:
#             A,B = balls[0],balls[1:]
#             for i in range(len(bins)):
#                 new_bins = copy.deepcopy(bins)
#                 new_bins[i].update({A:1})
#                 self._gen_helper(B,new_bins)
#
#     def get_all(self):
#         self._gen_helper(self.balls,self.bins)
#
# #BinPartitions(string.ascii_uppercase[:3],3).get_all()
#
# # below print all possibilities with empty cases:
# BinPartitions("0123456789",3).get_all()
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

# ##### code below reads from the text file above
#
# print("BREAK")
#
# ######## convert each line to list in python:
# a_file = open("examples/Coarse_Rob_fMNIST_tarAtt_cw/combos_10_3_string.txt", "r")
#
# list_of_lists = []
# for line in a_file:
#     stripped_line = line.strip()
#     letter_list = stripped_line.split(",")
#     list_temp = []
#     for item in letter_list:
#
#         a_list = item.split()
#         if len(a_list) > 0:
#             map_object = map(int, a_list[0])
#             list_of_integers = list(map_object)
#             list_temp.append(list_of_integers)
#         else:
#             list_temp.append([])
#     #line_list = stripped_line.split()
#     list_of_lists.append(list_temp)
#
# a_file.close()
#
# # np.save("examples/Coarse_Rob_fMNIST_tarAtt_cw/list_of_possibilities_of10_and_3.npy", list_of_lists)

print("break")


# ###################################################################################################
# #################################3 Greedy algorithm that considers all comibations of size Mc #####
# ###################################################################################################
# ######### loop over all non-empty cases possibilities (should be 55980, see ppt slides)
# list_of_all_possibilities = np.load("examples/Coarse_Rob_fMNIST_tarAtt_cw/list_of_possibilities_of10_and_3.npy", allow_pickle=True)
#
# max_PI = 0
# T_current_best = []
# for combo in list_of_all_possibilities:
#
#
#     # if combo does not conatain an empty list,
#     empty_list_case_flag = 0
#     for item in combo:
#         if len(item) == 0:
#             empty_list_case_flag = 1
#
#     if empty_list_case_flag == 0:
#
#         #########B calculate C_R, C_D, PI_1, PI_sum_metric,  flag_of_less_than_zero_element
#         _, _, PI_, PI_sum_metric, flag_of_less_than_zero_element = C_PI_calculator(
#             R_2, Delta_2, combo, 3)
#
#         if flag_of_less_than_zero_element == 0:
#
#             if PI_sum_metric > max_PI:
#                 max_PI = PI_sum_metric
#                 T_current_best = combo
#
#             print("Mapping = ", combo, "PI_sum_metric = ", [PI_sum_metric], "; 0 PI entry flag = ", flag_of_less_than_zero_element, '; current max is = ', max_PI, "; at mapping: ", T_current_best)
#
#
#
# print("break")




#################################################################################################################################
# ############################# below code only gives the cobinations; its copied to a text file "combos_10_3_string.txt"; Mc = 4
#################################################################################################################################
#!/usr/bin/python3
import string
import copy
#
# This generates all the possible placements of
# balls into boxes (configurations with empty boxes are allowed).
#
# class BinPartitions:
#
#     def __init__(self, balls, num_bins):
#         self.balls = balls
#         self.bins = [{} for x in range(num_bins)]
#
#     def print_bins(self, bins):
#         L = []
#         for b in bins:
#             buf = ''.join(sorted(b.keys()))
#             L += [buf]
#         with open("combos_3_2_string.txt","w") as f:
#             f.write(",".join(L))
#         print(",".join(L))
#
#     def _gen_helper(self,balls,bins):
#         if len(balls) == 0:
#             self.print_bins(bins)
#         else:
#             A,B = balls[0],balls[1:]
#             for i in range(len(bins)):
#                 new_bins = copy.deepcopy(bins)
#                 new_bins[i].update({A:1})
#                 self._gen_helper(B,new_bins)
#
#     def get_all(self):
#         self._gen_helper(self.balls,self.bins)

#BinPartitions(string.ascii_uppercase[:3],3).get_all()

# below print all possibilities with empty cases:
#BinPartitions("0123456789",4).get_all()
#BinPartitions("012",2).get_all()
#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

# ##### code below reads from the text file above
#
# print("BREAK")
#
# ######## convert each line to list in python:
# a_file = open("examples/Coarse_Rob_fMNIST_tarAtt_cw/combos_10_3_string.txt", "r")
#
# list_of_lists = []
# for line in a_file:
#     stripped_line = line.strip()
#     letter_list = stripped_line.split(",")
#     list_temp = []
#     for item in letter_list:
#
#         a_list = item.split()
#         if len(a_list) > 0:
#             map_object = map(int, a_list[0])
#             list_of_integers = list(map_object)
#             list_temp.append(list_of_integers)
#         else:
#             list_temp.append([])
#     #line_list = stripped_line.split()
#     list_of_lists.append(list_temp)
#
# a_file.close()
#
# # np.save("examples/Coarse_Rob_fMNIST_tarAtt_cw/list_of_possibilities_of10_and_3.npy", list_of_lists)

print("break")


# ##################################################################################################
# ################################3 Greedy algorithm that considers all comibations of size Mc=3 and M=10#####
# ##################################################################################################
# ######### loop over all non-empty cases possibilities (should be ~9000, see ppt slides)
# list_of_all_possibilities = np.load("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/list_of_possibilities_of10_and_4.npy", allow_pickle=True)
# M_c = 4
# max_PI = 0
# T_current_best = []
# idx = 0
# for combo in list_of_all_possibilities:
#     idx = idx + 1
#
#     # if combo does not conatain an empty list,
#     empty_list_case_flag = 0
#     for item in combo:
#         if len(item) == 0:
#             empty_list_case_flag = 1
#
#     if empty_list_case_flag == 0:
#
#         #########B calculate C_R, C_D, PI_1, PI_sum_metric,  flag_of_less_than_zero_element
#         _, _, PI_, PI_sum_metric, flag_of_less_than_zero_element = C_PI_calculator(
#             R_2, Delta_2, combo, M_c)
#
#         if flag_of_less_than_zero_element == 0:
#
#             if PI_sum_metric > max_PI:
#                 max_PI = PI_sum_metric
#                 T_current_best = combo
#
#             print("index", idx, "Mapping = ", combo, "PI_sum_metric = ", [PI_sum_metric], "; 0 PI entry flag = ", flag_of_less_than_zero_element, '; current max is = ', max_PI, "; at mapping: ", T_current_best)
#
#
#
# print("break")
#



