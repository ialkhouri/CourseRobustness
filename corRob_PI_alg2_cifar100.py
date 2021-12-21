import numpy
import numpy as np
import itertools
import random


############## This code is used for the local search procedure in the supplementary material and for running the brute force method


######## These constraints define the output specification for the coarse verification problem
import scipy.cluster.vq


########################### this is for cifar-100
save_list = np.load("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/cifar100_coarse_Rob_over_all_save_list.npy", allow_pickle=True)

save_list_2 = np.load("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/cifar100_coarse_Rob_over_all_save_list_2.npy", allow_pickle=True)



# save_list = save_list[:,0:4]
# temp = save_list[:,1]
# for index in range(len(temp)):
#     temp[index] = temp[index][0]
# save_list[:,1] = temp

####### define here a dictionary and define the values as lists
dict_data_1 = {}
dict_data_2 = {}
for idx in range(len(save_list)):
    item_1 = save_list[idx]
    #item_2 = save_list_2[idx]
    dict_data_1[((item_1[1]), item_1[2])] = []
    #dict_data_2[((item_2[1]), item_2[2])] = []

for idx in range(len(save_list_2)):
    #item_1 = save_list[idx]
    item_2 = save_list_2[idx]
    #dict_data_1[((item_1[1]), item_1[2])] = []
    dict_data_2[((item_2[1]), item_2[2])] = []


####### append to the list
for idx in range(len(save_list)):
    item_1 = save_list[idx]
    #item_2 = save_list_2[idx]
    dict_data_1[((item_1[1]), item_1[2])].append(item_1[3])
    #dict_data_2[((item_2[1]), item_2[2])].append(item_2[3])

for idx in range(len(save_list_2)):
    #item_1 = save_list[idx]
    item_2 = save_list_2[idx]
    #dict_data_1[((item_1[1]), item_1[2])].append(item_1[3])
    dict_data_2[((item_2[1]), item_2[2])].append(item_2[3])


##################### from these dictinoaries, find the fail ones and try to approximate
failed_pair_dict_1 = []
failed_pair_dict_2 = []
failed_pair_dict_2.append([8,33])
for i in range(100):
    for j in range(100):
        if i != j:
            if (i,j) not in dict_data_1.keys():
                #print('Fail indicies for dict_1 is found at ', [i,j])
                failed_pair_dict_1.append([i,j])
            if (i,j) not in dict_data_2.keys():
                #print('Fail indicies for dict_2 is found at ', [i,j])
                failed_pair_dict_2.append([i, j])


####### in the failed pairs, assign a value of 5 to L2 in each dictionary
for item in failed_pair_dict_1:
    dict_data_1[((item[0]), item[1])] = [ 5.0]
for item in failed_pair_dict_2:
    dict_data_2[((item[0]), item[1])] = [5.0]





####### take the mean and std
dict_data_final = {}
for pair in dict_data_2.keys():
    dict_data_final[((pair[0]), pair[1])]=[np.mean([dict_data_1[(int(pair[0]), pair[1])],dict_data_2[(int(pair[0]), pair[1])]]),
                                           np.std([dict_data_1[(int(pair[0]), pair[1])],dict_data_2[(int(pair[0]), pair[1])]])]
    #print("pair ", [pair], 'is good')

# ####### convert to np array
# labels_strings = ["Tshirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "ankle-boot", "bag", "Sneaker"]
# labels_strings_np = np.array(labels_strings)

final_table_mean = np.zeros(shape=(100,100))
final_table_sttd = np.zeros(shape=(100,100))
for i in range(100):
    for j in range(100):
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

########################################################## this is the swaping from the website mapping

TT_website = [[4, 30, 55, 72, 95],
 [1, 32, 67, 73, 91],
 [54, 62, 70, 82, 92],
 [9, 10, 16, 28, 61],
 [0, 51, 53, 57, 83],
 [22, 39, 40, 86, 87],
 [5, 20, 25, 84, 94],
 [6, 7, 14, 18, 24],
 [3, 42, 43, 88, 97],
 [12, 17, 37, 68, 76],
 [23, 33, 49, 60, 71],
 [15, 19, 21, 31, 38],
 [34, 63, 64, 66, 75],
 [26, 45, 77, 79, 99],
 [2, 11, 35, 46, 98],
 [27, 29, 44, 78, 93],
 [36, 50, 65, 74, 80],
 [47, 52, 56, 59, 96],
 [8, 13, 48, 58, 90],
 [41, 69, 81, 85, 89]]

TT_website_dolphin = [[4, 55, 72, 95],
 [1, 32, 67, 73, 91, 30],
 [54, 62, 70, 82, 92],
 [9, 10, 16, 28, 61],
 [0, 51, 53, 57, 83],
 [22, 39, 40, 86, 87],
 [5, 20, 25, 84, 94],
 [6, 7, 14, 18, 24],
 [3, 42, 43, 88, 97],
 [12, 17, 37, 68, 76],
 [23, 33, 49, 60, 71],
 [15, 19, 21, 31, 38],
 [34, 63, 64, 66, 75],
 [26, 45, 77, 79, 99],
 [2, 11, 35, 46, 98],
 [27, 29, 44, 78, 93],
 [36, 50, 65, 74, 80],
 [47, 52, 56, 59, 96],
 [8, 13, 48, 58, 90],
 [41, 69, 81, 85, 89]]

# #
# TT_website_flat = [item for lst in TT_website for item in lst]
# storage = []
# for i in range(100):
#     # choose the number to be going around for 19 times
#     number_to_going_around = TT_website_flat[i]
#
#     # find the super index and index of the number_to_going_around to find the source list
#     position = [(ii, jj.index(number_to_going_around)) for ii, jj in enumerate(TT_website) if number_to_going_around in jj]
#     # we found the source list
#     source_list = TT_website[position[0][0]]
#
#     # loop 19 times for every list in TT_website that is not the source list
#     TT_website_to_loop = [lst for lst in TT_website if lst != source_list]
#     for j in range(19):
#         destination_list = TT_website_to_loop[j]
#
#         ### operation of removing from the source list and adding to the destination list
#         TT_website[position[0][0]].remove(number_to_going_around)
#         destination_list.append(number_to_going_around)
#         #print(np.array(TT_website))
#
#         intermediate_list = []
#         for lst in TT_website:
#             tempp = []
#             for item in lst:
#                 tempp.append(item)
#             intermediate_list.append(tempp)
#
#
#         storage.append(intermediate_list)
#
#         ### operation of putting things back
#         TT_website[position[0][0]].append(number_to_going_around)
#         destination_list.remove(number_to_going_around)
#
# print("break")
# #
# #
# #
# #######try the mappings and save only those where \alpa_2 is higher than the website.
#
# mapping_best_than_website = []
# current_max = 3658.94
# for mapping in storage:
#     _, _, PI_star, PI_sum_metric_star, flag_of_less_than_zero_element_star = C_PI_calculator(R_2, Delta_2, mapping, 20)
#
#     if PI_sum_metric_star > current_max:
#         mapping_best_than_website = mapping
#         current_max = PI_sum_metric_star
#         print('Current best is at ', [PI_sum_metric_star])


# # ##################################################################################################################################################
# # ##################################################################################################################################################
# # ##################################################################################################################################################
# # ############# fiding local best value starting from T_website
# #
# def trying_1900_OneLabel_Mapps(TT, current_best):
#     TT_website_flat = [item for lst in TT for item in lst]
#     storage = []
#     for i in range(100):
#         # choose the number to be going around for 19 times
#         number_to_going_around = TT_website_flat[i]
#
#         # find the super index and index of the number_to_going_around to find the source list
#         position = [(ii, jj.index(number_to_going_around)) for ii, jj in enumerate(TT) if
#                     number_to_going_around in jj]
#         # we found the source list
#         source_list = TT[position[0][0]]
#
#         # loop 19 times for every list in TT_website that is not the source list
#         TT_website_to_loop = [lst for lst in TT if lst != source_list]
#         for j in range(19):
#             destination_list = TT_website_to_loop[j]
#
#             ### operation of removing from the source list and adding to the destination list
#             TT[position[0][0]].remove(number_to_going_around)
#             destination_list.append(number_to_going_around)
#             # print(np.array(TT_website))
#
#             intermediate_list = []
#             for lst in TT:
#                 tempp = []
#                 for item in lst:
#                     tempp.append(item)
#                 intermediate_list.append(tempp)
#
#             storage.append(intermediate_list)
#
#             ### operation of putting things back
#             TT[position[0][0]].append(number_to_going_around)
#             destination_list.remove(number_to_going_around)
#
#     mapping_best_than_website = []
#     current_max = current_best
#     for mapping in storage:
#         # if the considered mapping does not have a one-label category, then don't even bother to try
#         if 1 not in [len(set_S) for set_S in mapping]:
#             _, _, PI_star, PI_sum_metric_star, flag_of_less_than_zero_element_star = C_PI_calculator(R_2, Delta_2, mapping,
#                                                                                                      20)
#
#             if PI_sum_metric_star > current_max:
#                 mapping_best_than_website = mapping
#                 current_max = PI_sum_metric_star
#                 print('Current best from ', current_best ,' is at ', [PI_sum_metric_star])
#
#     # if the current best mappoing stayed empty, then make it equal to the input of the function
#     if mapping_best_than_website == []:
#         mapping_best_than_website = TT
#         print("non of the tried combinations in the current batch returned a better mapping than the current")
#
#
#     return current_max,mapping_best_than_website
#
# # ### Below is to test the function:
# # current_max,mapping_best_than_website = trying_1900_OneLabel_Mapps(TT_website, 3658.9)
# # current_max_2,mapping_best_than_website_2 = trying_1900_OneLabel_Mapps(mapping_best_than_website, current_max)
#
#
# ################ here goes the while loop; keep looping for a very long time. the stopping criteria: (1) one label group, or (2) after number of tries exceeds some threshold
# ################ OR the stopping criteria can be if all the 1900 tries are tried and input and output are the same
# iteration_counter = 0
# current_mapping = TT_website
# current_value   = 3658.9
# while iteration_counter < 1000:
#     iteration_counter = iteration_counter + 1
# #while best_value > current_value:
#     best_value, best_mapping = trying_1900_OneLabel_Mapps(current_mapping, current_value)
#
#     if best_value == current_value:
#         print("Further enhancement is not possible. hence, we are exiting")
#         break
#     if best_value > current_value:
#         print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
#         print("ITERATION ", [iteration_counter], '[BEST, CURRENT]', [best_value , current_value])
#         print("###################################################################################")
#
#     current_value   = best_value
#     current_mapping = best_mapping
#
# np.save("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/cifar100_best_mapping_from_swaping_theWeb_mapping_extended_without_oneLblGroups.npy", best_mapping)
#
# print("break")


##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
############### IS THE CURRENT best driven from T website semantic based???


my_file = open("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/cifar_100_fine_label_names.txt", "r")
Actual_labels_list = [line.split(',') for line in my_file.readlines()]
my_file.close()

T_best_from_web = np.load("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/cifar100_best_mapping_from_swaping_theWeb_mapping_extended_without_oneLblGroups.npy", allow_pickle=True)

grouping_of_lables_best_from_website = []
for lst in list(T_best_from_web):
    temp = []
    for lbl in lst:
        temp.append(Actual_labels_list[lbl])
    grouping_of_lables_best_from_website.append(temp)

print("break")


# C_R_star, C_D_star, PI_star_from_website, PI_sum_metric_star, flag_of_less_than_zero_element_star = C_PI_calculator(     R_2, Delta_2, TT_website_dolphin, 20)
#
# np.save("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/cifar100_best_PI_1st_iteration.npy", [PI_star_from_website])

print("break")
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################


#
#
#
#
# print('break')
#
# import pandas as pd
#
# ## convert your array into a dataframe
# df = pd.DataFrame(PI_star).T
#
# ## save to xlsx file
#
# #filepath = 'cifar100_PI_from_cifar.xlsx'
#
# df.to_csv("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/cifar100_PI_from_cifar.csv",index=False)
#
# df.to_excel(excel_writer = "/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/cifar100_PI_from_cifar.xlsx")
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################

# ################ saving PI for random, web, best_from_web
#
# ######3 this is random
# # this is for M_c = 20
# M_c = 20
# ranom_list = random.sample(range(100), 100)
# #T_init = [ranom_list[0:3], ranom_list[3:5], ranom_list[5:7], ranom_list[7:10]]
# data_list = ranom_list
# i=0
# T_init=[]
# while i<len(data_list):
#   T_init.append(data_list[i:i+int(100/20)])
#   i+=int(100/20)
# #print(T_init)
# C_R_star, C_D_star, PI_star_random, PI_sum_metric_star, flag_of_less_than_zero_element_star = C_PI_calculator(
#     R_2, Delta_2, T_init, 20)


# C_R_star, C_D_star, PI_star_website, PI_sum_metric_star, flag_of_less_than_zero_element_star = C_PI_calculator(
#     R_2, Delta_2, TT_website, 20)
#
# T_best_from_web = np.load("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/cifar100_best_mapping_from_swaping_theWeb_mapping.npy", allow_pickle=True)
# C_R_star, C_D_star, PI_star_best_from_web, PI_sum_metric_star, flag_of_less_than_zero_element_star = C_PI_calculator(
#     R_2, Delta_2, T_best_from_web[1], 20)

# np.save("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/cifar100_random_PI_toShow.npy", [PI_star_random])
# np.save("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/cifar100_T_ini_random_toShow.npy", [T_init])


##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
############ reading the text for the 100 labels to get the random, website, best grouping w.r.t. to the actual labels for showing.


# my_file = open("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/cifar_100_fine_label_names.txt", "r")
# #content = my_file.read()
# #Actual_labels_list = content.split(",")
# Actual_labels_list = [line.split(',') for line in my_file.readlines()]
# my_file.close()
#
#
# ### [RANDOM] a list for actual labels from .txt file
# T_ini = np.load("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/cifar100_T_ini_random_toShow.npy", allow_pickle=True)
# grouping_of_lables_random = []
# for lst in list(T_ini[0]):
#     temp = []
#     for lbl in lst:
#         temp.append(Actual_labels_list[lbl])
#     grouping_of_lables_random.append(temp)
# np.save("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/grouping_of_lables_random.npy", grouping_of_lables_random)
#
# ### [WEBSITE] a list for actual labels from .txt file
# grouping_of_lables_WEBSITE = []
# for lst in TT_website:
#     temp = []
#     for lbl in lst:
#         temp.append(Actual_labels_list[lbl])
#     grouping_of_lables_WEBSITE.append(temp)
# np.save("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/grouping_of_lables_WEBSITE.npy", grouping_of_lables_WEBSITE)
#
# ### [BETTER THAN WEBSITE] a list for actual labels from .txt file
# T_best_from_web = np.load("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/cifar100_best_mapping_from_swaping_theWeb_mapping.npy", allow_pickle=True)
# grouping_of_lables_BEST_FROM_WEBSITE = []
# for lst in T_best_from_web[1]:
#     temp = []
#     for lbl in lst:
#         temp.append(Actual_labels_list[lbl])
#     grouping_of_lables_BEST_FROM_WEBSITE.append(temp)
# np.save("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/grouping_of_lables_BEST_FROM_WEBSITE.npy", grouping_of_lables_BEST_FROM_WEBSITE)
#
# print("break")






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
# # # this is for M_c = 3
# # ranom_list = random.sample(range(10), 10)
# # T_init = [ranom_list[0:3], ranom_list[3:5], ranom_list[5:10]]
#
# # # this is for M_c = 4
# # ranom_list = random.sample(range(10), 10)
# # T_init = [ranom_list[0:3], ranom_list[3:5], ranom_list[5:7], ranom_list[7:10]]
# #
#
# # this is for M_c = 20
# M_c = 20
# ranom_list = random.sample(range(100), 100)
# #T_init = [ranom_list[0:3], ranom_list[3:5], ranom_list[5:7], ranom_list[7:10]]
# data_list = ranom_list
# i=0
# T_init=[]
# while i<len(data_list):
#   T_init.append(data_list[i:i+int(100/20)])
#   i+=int(100/20)
# #print(T_init)
#
#
#
#
# # FLAG If the initial mapping_T has empty set
# for ls in T_init:
#     if ls == []:
#         print("WE HAVE THE PROBLEM OF AN EMPTY GROUP,  FIX", T_init)
#
# ###################### If the maximum is reapeated, exit with best mapping
# number_or_iterations = 100000
# mappings_list_to_save = []
# sum_ofPI_toSave       = []
# bad_mappings_list_to_save = []
# bad_sum_ofPI_toSave = []
# ######A start with T = T_spec_clustering with flag = 0
# T_current = T_init
#
# M_c = 20
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
#     # if current_best_mapping_based_onSumOfPI is bigger than PI_sum_metric_current, then update current
#     if best_mapping_based_onSumOfPI < PI_sum_metric_current and flag_of_less_than_zero_element == "Feasible":
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
#         M_c = 20
#         ranom_list = random.sample(range(100), 100)
#         # T_init = [ranom_list[0:3], ranom_list[3:5], ranom_list[5:7], ranom_list[7:10]]
#         data_list = ranom_list
#         i = 0
#         T_current_new = []
#         while i < len(data_list):
#             T_current_new.append(data_list[i:i + int(100 / M_c)])
#             i += int(100 / M_c)
#
#     ######## if a current bad mapping is repeated, then restart T_current
#     #if T_current_new in bad_mappings_list_to_save and flag_of_less_than_zero_element_current == 1:
#     if PI_sum_metric_current in bad_sum_ofPI_toSave and flag_of_less_than_zero_element_current == 1:
#         #print("Restarting T_current randomly because of a non-feasible value is repeated ")
#         restart = "Yes - Non-Feasible is repeated"
#         M_c = 20
#         ranom_list = random.sample(range(100), 100)
#         # T_init = [ranom_list[0:3], ranom_list[3:5], ranom_list[5:7], ranom_list[7:10]]
#         data_list = ranom_list
#         i = 0
#         T_current_new = []
#         while i < len(data_list):
#             T_current_new.append(data_list[i:i + int(100 / M_c)])
#             i += int(100 / M_c)
#
#
#     ####### logger:
#     #print("Iteration: ", iter, "current best value = ", [best_mapping_based_onSumOfPI] , "; Current and Next mappings are ", [T_current], " ===> ", [T_current_new] , '; STATUS = ', flag_of_less_than_zero_element, "; sum of PI metric = ", [PI_sum_metric_current], "; moving label ", [m_source], " to ", [S_dest])
#
#     ####### logger:
#     #print("Iteration: ", iter, "current best value = ", [best_mapping_based_onSumOfPI] , "; Current mapping is ", [T_current], '; STATUS = ', flag_of_less_than_zero_element, "; sum of PI metric = ", [PI_sum_metric_current], "Restart = ", restart)
#
#     ####### logger:
#     print("Iteration: ", iter, "current best value = ", [best_mapping_based_onSumOfPI] , '; STATUS = ', flag_of_less_than_zero_element, "; sum of PI metric = ", [PI_sum_metric_current], "Restart = ", restart)
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
#
#
#
# ############################################################################################################################################################################################################
# ############################################################################################################################################################################################################
# ############################################################################################################################################################################################################
# #######################################################################################################################################
# ############################################################################################################################################################################################################
# ############################################################################################################################################################################################################
# ############################################################################################################################################################################################################
#
#
#
#
# #################################################################################################################################
# # ############################# below code only gives the cobinations; its copied to a text file "combos_10_3_string.txt"; Mc = 3
# #################################################################################################################################
# # #!/usr/bin/python3
# # import string
# # import copy
# # #
# # # This generates all the possible placements of
# # # balls into boxes (configurations with empty boxes are allowed).
# # #
# # class BinPartitions:
# #
# #     def __init__(self, balls, num_bins):
# #         self.balls = balls
# #         self.bins = [{} for x in range(num_bins)]
# #
# #     def print_bins(self, bins):
# #         L = []
# #         for b in bins:
# #             buf = ''.join(sorted(b.keys()))
# #             L += [buf]
# #         print(",".join(L))
# #
# #     def _gen_helper(self,balls,bins):
# #         if len(balls) == 0:
# #             self.print_bins(bins)
# #         else:
# #             A,B = balls[0],balls[1:]
# #             for i in range(len(bins)):
# #                 new_bins = copy.deepcopy(bins)
# #                 new_bins[i].update({A:1})
# #                 self._gen_helper(B,new_bins)
# #
# #     def get_all(self):
# #         self._gen_helper(self.balls,self.bins)
# #
# # #BinPartitions(string.ascii_uppercase[:3],3).get_all()
# #
# # # below print all possibilities with empty cases:
# # BinPartitions("0123456789",3).get_all()
# #################################################################################################################################
# #################################################################################################################################
# #################################################################################################################################
#
# # ##### code below reads from the text file above
# #
# # print("BREAK")
# #
# # ######## convert each line to list in python:
# # a_file = open("examples/Coarse_Rob_fMNIST_tarAtt_cw/combos_10_3_string.txt", "r")
# #
# # list_of_lists = []
# # for line in a_file:
# #     stripped_line = line.strip()
# #     letter_list = stripped_line.split(",")
# #     list_temp = []
# #     for item in letter_list:
# #
# #         a_list = item.split()
# #         if len(a_list) > 0:
# #             map_object = map(int, a_list[0])
# #             list_of_integers = list(map_object)
# #             list_temp.append(list_of_integers)
# #         else:
# #             list_temp.append([])
# #     #line_list = stripped_line.split()
# #     list_of_lists.append(list_temp)
# #
# # a_file.close()
# #
# # # np.save("examples/Coarse_Rob_fMNIST_tarAtt_cw/list_of_possibilities_of10_and_3.npy", list_of_lists)
#
# print("break")
#
#
# # ###################################################################################################
# # #################################3 Greedy algorithm that considers all comibations of size Mc #####
# # ###################################################################################################
# # ######### loop over all non-empty cases possibilities (should be 55980, see ppt slides)
# # list_of_all_possibilities = np.load("examples/Coarse_Rob_fMNIST_tarAtt_cw/list_of_possibilities_of10_and_3.npy", allow_pickle=True)
# #
# # max_PI = 0
# # T_current_best = []
# # for combo in list_of_all_possibilities:
# #
# #
# #     # if combo does not conatain an empty list,
# #     empty_list_case_flag = 0
# #     for item in combo:
# #         if len(item) == 0:
# #             empty_list_case_flag = 1
# #
# #     if empty_list_case_flag == 0:
# #
# #         #########B calculate C_R, C_D, PI_1, PI_sum_metric,  flag_of_less_than_zero_element
# #         _, _, PI_, PI_sum_metric, flag_of_less_than_zero_element = C_PI_calculator(
# #             R_2, Delta_2, combo, 3)
# #
# #         if flag_of_less_than_zero_element == 0:
# #
# #             if PI_sum_metric > max_PI:
# #                 max_PI = PI_sum_metric
# #                 T_current_best = combo
# #
# #             print("Mapping = ", combo, "PI_sum_metric = ", [PI_sum_metric], "; 0 PI entry flag = ", flag_of_less_than_zero_element, '; current max is = ', max_PI, "; at mapping: ", T_current_best)
# #
# #
# #
# # print("break")
#
#
#
#
# #################################################################################################################################
# # ############################# below code only gives the cobinations; its copied to a text file "combos_10_3_string.txt"; Mc = 4
# #################################################################################################################################
# #!/usr/bin/python3
# import string
# import copy
# #
# # This generates all the possible placements of
# # balls into boxes (configurations with empty boxes are allowed).
# #
# # class BinPartitions:
# #
# #     def __init__(self, balls, num_bins):
# #         self.balls = balls
# #         self.bins = [{} for x in range(num_bins)]
# #
# #     def print_bins(self, bins):
# #         L = []
# #         for b in bins:
# #             buf = ''.join(sorted(b.keys()))
# #             L += [buf]
# #         with open("combos_3_2_string.txt","w") as f:
# #             f.write(",".join(L))
# #         print(",".join(L))
# #
# #     def _gen_helper(self,balls,bins):
# #         if len(balls) == 0:
# #             self.print_bins(bins)
# #         else:
# #             A,B = balls[0],balls[1:]
# #             for i in range(len(bins)):
# #                 new_bins = copy.deepcopy(bins)
# #                 new_bins[i].update({A:1})
# #                 self._gen_helper(B,new_bins)
# #
# #     def get_all(self):
# #         self._gen_helper(self.balls,self.bins)
#
# #BinPartitions(string.ascii_uppercase[:3],3).get_all()
#
# # below print all possibilities with empty cases:
# #BinPartitions("0123456789",4).get_all()
# #BinPartitions("012",2).get_all()
# #################################################################################################################################
# #################################################################################################################################
# #################################################################################################################################
#
# # ##### code below reads from the text file above
# #
# # print("BREAK")
# #
# # ######## convert each line to list in python:
# # a_file = open("examples/Coarse_Rob_fMNIST_tarAtt_cw/combos_10_3_string.txt", "r")
# #
# # list_of_lists = []
# # for line in a_file:
# #     stripped_line = line.strip()
# #     letter_list = stripped_line.split(",")
# #     list_temp = []
# #     for item in letter_list:
# #
# #         a_list = item.split()
# #         if len(a_list) > 0:
# #             map_object = map(int, a_list[0])
# #             list_of_integers = list(map_object)
# #             list_temp.append(list_of_integers)
# #         else:
# #             list_temp.append([])
# #     #line_list = stripped_line.split()
# #     list_of_lists.append(list_temp)
# #
# # a_file.close()
# #
# # # np.save("examples/Coarse_Rob_fMNIST_tarAtt_cw/list_of_possibilities_of10_and_3.npy", list_of_lists)
#
# print("break")
#
#
# # ##################################################################################################
# # ################################3 Greedy algorithm that considers all comibations of size Mc=3 and M=10#####
# # ##################################################################################################
# # ######### loop over all non-empty cases possibilities (should be ~9000, see ppt slides)
# # list_of_all_possibilities = np.load("/home/ismail/pycharmProjects/SSLTL_project/Coarse_Robust/list_of_possibilities_of10_and_4.npy", allow_pickle=True)
# # M_c = 4
# # max_PI = 0
# # T_current_best = []
# # idx = 0
# # for combo in list_of_all_possibilities:
# #     idx = idx + 1
# #
# #     # if combo does not conatain an empty list,
# #     empty_list_case_flag = 0
# #     for item in combo:
# #         if len(item) == 0:
# #             empty_list_case_flag = 1
# #
# #     if empty_list_case_flag == 0:
# #
# #         #########B calculate C_R, C_D, PI_1, PI_sum_metric,  flag_of_less_than_zero_element
# #         _, _, PI_, PI_sum_metric, flag_of_less_than_zero_element = C_PI_calculator(
# #             R_2, Delta_2, combo, M_c)
# #
# #         if flag_of_less_than_zero_element == 0:
# #
# #             if PI_sum_metric > max_PI:
# #                 max_PI = PI_sum_metric
# #                 T_current_best = combo
# #
# #             print("index", idx, "Mapping = ", combo, "PI_sum_metric = ", [PI_sum_metric], "; 0 PI entry flag = ", flag_of_less_than_zero_element, '; current max is = ', max_PI, "; at mapping: ", T_current_best)
# #
# #
# #
# # print("break")
#
#
#
#
