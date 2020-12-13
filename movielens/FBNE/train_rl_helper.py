__author__ = 'haobowen'

import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import setting
from time import time
import logging
import torch
import tqdm
from scipy.stats import pearsonr
import gendata_fbne as data
import scipy.sparse as sp
import metrics as metrics
import heapq
import tensorflow.contrib.layers as layers

import tensorflow as tf
from FBNEConv import GeneralGNN


class environment():
    def __init__(self):

        self.original_user_ebd, self.original_item_ebd = self.parse_original_embeddings()
        self.batch_size = setting.batch_size
        self.embedding_size = setting.embedding_size
        self.padding_number_user_task = setting.num_users
        self.padding_number_item_task = setting.num_items


    def assign_batch_data(self, train_batches):
        self.train_batches = train_batches

    def parse_train_batches(self, train_batches, batch_index):
        return [(train_batches[r])[batch_index] for r in range(9)]

    def parse_original_embeddings(self):
        return np.load(setting.original_user_ebd), np.load(setting.original_item_ebd)

    def initilize_state_user_task(self, max_num, state_size, batch_original_reward, original_reward):
        self.max_num = max_num
        self.batch_original_reward = batch_original_reward
        self.original_reward = original_reward
        self.origin_prob = np.zeros((self.batch_size, 1), dtype=np.float32)  
        self.dot_product = np.zeros((self.batch_size, 1), dtype=np.float32) 
        self.element_wise_current_mean1 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum1 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_mean2 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum2 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_mean3 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum3 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.vector_current_sum1 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.vector_current_mean1 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.num_selected = np.zeros(self.batch_size, dtype=np.int)
        self.action_matrix = np.zeros((self.batch_size, self.max_num), dtype=np.int)
        self.state_matrix = np.zeros((self.batch_size, self.max_num, state_size), dtype=np.float32)
        self.selected_input = np.full((self.batch_size, self.max_num), self.padding_number_user_task)

    def initilize_state_user_task_3rd(self, max_num_3rd, state_size_3rd):
        self.max_num_3rd = max_num_3rd
        self.origin_prob_3rd = np.zeros((self.batch_size, 1), dtype=np.float32)  
        self.dot_product_3rd = np.zeros((self.batch_size, 1), dtype=np.float32) 
        self.element_wise_current_mean1_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum1_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_mean2_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum2_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_mean3_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum3_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_mean4_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum4_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_mean5_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum5_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.vector_current_sum1_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.vector_current_mean1_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.num_selected_3rd = np.zeros(self.batch_size, dtype=np.int)
        self.action_matrix_3rd = np.zeros((self.batch_size, self.max_num_3rd), dtype=np.int)
        self.state_matrix_3rd = np.zeros((self.batch_size, self.max_num_3rd, state_size_3rd), dtype=np.float32)
        self.selected_input_3rd = np.full((self.batch_size, self.max_num_3rd), self.padding_number_user_task)

    def get_state_user_task(self, batch_prob1, batch_prob2, batch_target_ori_ebd, batch_target_aggre_ebd,
                            batch_first_order_ori_ebd, batch_second_neigh_ori_ebd, step_index):


        self.first_order_prob1 = np.reshape(batch_prob1, (-1, 1))  # [b, 1] only first aggregator
        self.first_and_second_oreer_prob2 = np.reshape(batch_prob2, (-1, 1))  # [b, 1] first and second aggrgator
        self.cos_sim1 = self.cos_sim(batch_target_ori_ebd, batch_second_neigh_ori_ebd, step_index)
        self.cos_sim2 = self.cos_sim(batch_target_aggre_ebd, batch_second_neigh_ori_ebd, step_index)
        self.cos_sim3 = self.cos_sim(np.mean(batch_first_order_ori_ebd, 1), batch_second_neigh_ori_ebd,
                                     step_index)
        self.element_wise_current1 = self.cal_element_wise(batch_target_ori_ebd, batch_second_neigh_ori_ebd, step_index)
        self.element_wise_current2 = self.cal_element_wise(batch_target_aggre_ebd, batch_second_neigh_ori_ebd,
                                                           step_index)
        self.element_wise_current3 = self.cal_element_wise(np.mean(batch_first_order_ori_ebd, 1),
                                                           batch_second_neigh_ori_ebd, step_index)
        self.vector_current1 = batch_target_ori_ebd  # [b, e]
        self.vector_current2 = batch_target_aggre_ebd  # [b, e]
        self.vector_current3 = np.mean(batch_first_order_ori_ebd, 1)  # [b, n1, e] -> [b, e]
        self.vector_current4 = batch_second_neigh_ori_ebd[:, step_index, :]  # [b, n2, e] extract-> [b, e]


        return np.concatenate(
            (self.cos_sim1, self.cos_sim2, self.cos_sim3,
             self.element_wise_current1, self.element_wise_current2, self.element_wise_current3,
             self.element_wise_current_mean1, self.element_wise_current_mean2, self.element_wise_current_mean3,
             self.vector_current1, self.vector_current2, self.vector_current3, self.vector_current4,
             self.vector_current_mean1), 1)

    def get_state_user_task_3rd(self, batch_target_ori_ebd, batch_target_aggre_ebd,
                                batch_target_second_aggre_ebd,
                                batch_first_neigh_ori_ebd, batch_second_neigh_ori_ebd,
                                batch_third_neigh_ori_ebd, step_index):

        self.cos_sim1_3rd = self.cos_sim(batch_target_ori_ebd, batch_third_neigh_ori_ebd, step_index)

        self.cos_sim2_3rd = self.cos_sim(np.mean(batch_first_neigh_ori_ebd, 1), batch_third_neigh_ori_ebd,
                                         step_index)
        self.cos_sim3_3rd = self.cos_sim(np.mean(batch_second_neigh_ori_ebd, 1), batch_third_neigh_ori_ebd, step_index)

        self.cos_sim4_3rd = self.cos_sim(batch_target_aggre_ebd, batch_third_neigh_ori_ebd, step_index)
        self.cos_sim5_3rd = self.cos_sim(batch_target_second_aggre_ebd, batch_third_neigh_ori_ebd, step_index)

        self.element_wise_current1_3rd = self.cal_element_wise(batch_target_ori_ebd, batch_third_neigh_ori_ebd,
                                                               step_index)
        self.element_wise_current2_3rd = self.cal_element_wise(np.mean(batch_first_neigh_ori_ebd, 1),
                                                               batch_third_neigh_ori_ebd, step_index)
        self.element_wise_current3_3rd = self.cal_element_wise(np.mean(batch_second_neigh_ori_ebd, 1),
                                                               batch_third_neigh_ori_ebd, step_index)
        self.element_wise_current4_3rd = self.cal_element_wise(batch_target_aggre_ebd, batch_third_neigh_ori_ebd,
                                                               step_index)
        self.element_wise_current5_3rd = self.cal_element_wise(batch_target_second_aggre_ebd, batch_third_neigh_ori_ebd,
                                                               step_index)

        self.vector_current1_3rd = batch_target_ori_ebd  # [b, e]
        self.vector_current2_3rd = np.mean(batch_first_neigh_ori_ebd, 1)  # [b, n1, e] -> [b, e]
        self.vector_current3_3rd = np.mean(batch_second_neigh_ori_ebd, 1)
        self.vector_current4_3rd = batch_third_neigh_ori_ebd[:, step_index, :]
        self.vector_current5_3rd = batch_target_aggre_ebd  # [b, e]
        self.vector_current6_3rd = batch_target_second_aggre_ebd

        return np.concatenate((self.cos_sim1_3rd, self.cos_sim2_3rd, self.cos_sim3_3rd, self.cos_sim4_3rd,
                               self.cos_sim5_3rd, self.element_wise_current1_3rd,
                               self.element_wise_current2_3rd, self.element_wise_current3_3rd,
                               self.element_wise_current4_3rd, self.element_wise_current5_3rd,
                               self.vector_current1_3rd, self.vector_current2_3rd, self.vector_current3_3rd,
                               self.vector_current4_3rd, self.vector_current5_3rd,
                               self.vector_current6_3rd), 1)

    def initilize_state_item_task(self, max_num, state_size, batch_original_reward, original_reward):
        self.max_num = max_num
        self.batch_original_reward = batch_original_reward
        self.original_reward = original_reward
        self.origin_prob = np.zeros((self.batch_size, 1), dtype=np.float32)  
        self.dot_product = np.zeros((self.batch_size, 1), dtype=np.float32)
        self.element_wise_current_mean1 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum1 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_mean2 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum2 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_mean3 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum3 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.vector_current_sum1 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.vector_current_mean1 = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.num_selected = np.zeros(self.batch_size, dtype=np.int)
        self.action_matrix = np.zeros((self.batch_size, self.max_num), dtype=np.int)
        self.state_matrix = np.zeros((self.batch_size, self.max_num, state_size), dtype=np.float32)
        self.selected_input = np.full((self.batch_size, self.max_num), self.padding_number_item_task)

    def initilize_state_item_task_3rd(self, max_num_3rd, state_size_3rd):
        self.max_num_3rd = max_num_3rd
        self.element_wise_current_mean1_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum1_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_mean2_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum2_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_mean3_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum3_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_mean4_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum4_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_mean5_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.element_wise_current_sum5_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.vector_current_sum1_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.vector_current_mean1_3rd = np.zeros((self.batch_size, self.embedding_size), dtype=np.float32)
        self.num_selected_3rd = np.zeros(self.batch_size, dtype=np.int)
        self.action_matrix_3rd = np.zeros((self.batch_size, self.max_num_3rd), dtype=np.int)
        self.state_matrix_3rd = np.zeros((self.batch_size, self.max_num_3rd, state_size_3rd), dtype=np.float32)
        self.selected_input_3rd = np.full((self.batch_size, self.max_num_3rd), self.padding_number_item_task)

    def get_state_item_task(self, batch_prob1, batch_prob2, batch_target_ori_ebd, batch_target_aggre_ebd,
                            batch_first_order_ori_ebd, batch_second_neigh_ori_ebd, step_index):

        # probability
        self.first_order_prob1 = np.reshape(batch_prob1, (-1, 1))  # [b, 1] only first aggregator
        self.first_and_second_oreer_prob2 = np.reshape(batch_prob2, (-1, 1))  # [b, 1] first and second aggrgator

        # cosine similarity
        self.cos_sim1 = self.cos_sim(batch_target_ori_ebd, batch_second_neigh_ori_ebd, step_index)
        self.cos_sim2 = self.cos_sim(batch_target_aggre_ebd, batch_second_neigh_ori_ebd, step_index)
        self.cos_sim3 = self.cos_sim(np.mean(batch_first_order_ori_ebd, 1), batch_second_neigh_ori_ebd,
                                     step_index)

        # element wise
        self.element_wise_current1 = self.cal_element_wise(batch_target_ori_ebd, batch_second_neigh_ori_ebd, step_index)
        self.element_wise_current2 = self.cal_element_wise(batch_target_aggre_ebd, batch_second_neigh_ori_ebd,
                                                           step_index)
        self.element_wise_current3 = self.cal_element_wise(np.mean(batch_first_order_ori_ebd, 1),
                                                           batch_second_neigh_ori_ebd, step_index)

        # current
        self.vector_current1 = batch_target_ori_ebd  # [b, e]
        self.vector_current2 = batch_target_aggre_ebd  # [b, e]
        self.vector_current3 = np.mean(batch_first_order_ori_ebd, 1)  # [b, n1, e] -> [b, e]
        self.vector_current4 = batch_second_neigh_ori_ebd[:, step_index, :]  # [b, n2, e] extract-> [b, e]

        return np.concatenate(
            (self.cos_sim1, self.cos_sim2, self.cos_sim3,
             self.element_wise_current1, self.element_wise_current2, self.element_wise_current3,
             self.element_wise_current_mean1, self.element_wise_current_mean2, self.element_wise_current_mean3,
             self.vector_current1, self.vector_current2, self.vector_current3, self.vector_current4,
             self.vector_current_mean1), 1)

    def get_state_item_task_3rd(self, batch_target_ori_ebd, batch_target_aggre_ebd, batch_target_second_aggre_ebd,
                                batch_first_neigh_ori_ebd, batch_second_neigh_ori_ebd, batch_third_neigh_ori_ebd,
                                step_index):

        self.cos_sim1_3rd = self.cos_sim(batch_target_ori_ebd, batch_third_neigh_ori_ebd, step_index)

        self.cos_sim2_3rd = self.cos_sim(np.mean(batch_first_neigh_ori_ebd, 1), batch_third_neigh_ori_ebd,
                                         step_index)
        self.cos_sim3_3rd = self.cos_sim(np.mean(batch_second_neigh_ori_ebd, 1), batch_third_neigh_ori_ebd, step_index)

        self.cos_sim4_3rd = self.cos_sim(batch_target_aggre_ebd, batch_third_neigh_ori_ebd, step_index)
        self.cos_sim5_3rd = self.cos_sim(batch_target_second_aggre_ebd, batch_third_neigh_ori_ebd, step_index)

        # element wise
        self.element_wise_current1_3rd = self.cal_element_wise(batch_target_ori_ebd, batch_third_neigh_ori_ebd,
                                                               step_index)
        self.element_wise_current2_3rd = self.cal_element_wise(np.mean(batch_first_neigh_ori_ebd, 1),
                                                               batch_third_neigh_ori_ebd, step_index)
        self.element_wise_current3_3rd = self.cal_element_wise(np.mean(batch_second_neigh_ori_ebd, 1),
                                                               batch_third_neigh_ori_ebd, step_index)
        self.element_wise_current4_3rd = self.cal_element_wise(batch_target_aggre_ebd, batch_third_neigh_ori_ebd,
                                                               step_index)
        self.element_wise_current5_3rd = self.cal_element_wise(batch_target_second_aggre_ebd, batch_third_neigh_ori_ebd,
                                                               step_index)

        # current
        self.vector_current1_3rd = batch_target_ori_ebd  # [b, e]
        self.vector_current2_3rd = np.mean(batch_first_neigh_ori_ebd, 1)  # [b, n1, e] -> [b, e]
        self.vector_current3_3rd = np.mean(batch_second_neigh_ori_ebd, 1)
        self.vector_current4_3rd = batch_third_neigh_ori_ebd[:, step_index, :]
        self.vector_current5_3rd = batch_target_aggre_ebd  # [b, e]
        self.vector_current6_3rd = batch_target_second_aggre_ebd

        return np.concatenate((self.cos_sim1_3rd, self.cos_sim2_3rd, self.cos_sim3_3rd, self.cos_sim4_3rd,
                               self.cos_sim5_3rd, self.element_wise_current1_3rd,
                               self.element_wise_current2_3rd, self.element_wise_current3_3rd,
                               self.element_wise_current4_3rd, self.element_wise_current5_3rd,
                               self.vector_current1_3rd, self.vector_current2_3rd, self.vector_current3_3rd,
                               self.vector_current4_3rd, self.vector_current5_3rd,
                               self.vector_current6_3rd), 1)

    def get_selected_second_order_neighbors_item_task(self, b_second_order_item):
        row_deleted_index = []

        # define row_deleted_flag, if each value equals to 0, it means delete, if each value equals to 1, it mains maintain
        row_deleted_flag = np.ones((self.batch_size), dtype=np.int)

        selected_input = np.zeros((self.batch_size, self.max_num), dtype=np.int)
        for batch_index in range(self.batch_size):
            selected = []
            for neighbor_index in range(self.max_num):
                if self.action_matrix[batch_index, neighbor_index] == 1:
                    selected.append(b_second_order_item[batch_index, neighbor_index])
            # random select one course from the original enrolled courses if no course is selected by the agent, change the number of selected courses as 1 at the same time
            if len(selected) == 0:
                row_deleted_flag[batch_index] = 1

                row_deleted_index.append(batch_index)
                original_second_order_neghbors = list(set(b_second_order_item[batch_index]))
                if self.padding_number_item_task in original_second_order_neghbors:
                    original_second_order_neghbors.remove(self.padding_number_item_task)
                random_neighbor = np.random.choice(original_second_order_neghbors, 1)[0]
                selected.append(random_neighbor)
                self.num_selected[batch_index] = 1

            for neighbor_index in range(self.max_num - len(selected)):
                selected.append(self.padding_number_item_task)
            selected_input[batch_index, :] = np.array(selected)
        return selected_input, row_deleted_index, row_deleted_flag

    def get_selected_third_order_neighbors_item_task(self, b_third_order_users, row_deleted_flag_2rd):
        row_deleted_index = []
        selected_input = np.zeros((self.batch_size, self.max_num_3rd), dtype=np.int)
        for batch_index in range(self.batch_size):
            selected = []
            for neighbor_index in range(self.max_num_3rd):
                if self.action_matrix_3rd[batch_index, neighbor_index] == 1:
                    selected.append(b_third_order_users[batch_index, neighbor_index])
            if len(selected) == 0:
                row_deleted_index.append(batch_index)
                original_third_order_neighbors = list(set(b_third_order_users[batch_index]))
                if self.padding_number_user_task in original_third_order_neighbors:
                    original_third_order_neighbors.remove(self.padding_number_user_task)
                random_neighbor = np.random.choice(original_third_order_neighbors, 1)[0]
                selected.append(random_neighbor)

            for neighbor_index in range(self.max_num_3rd - len(selected)):
                selected.append(self.padding_number_user_task)
            selected_input[batch_index, :] = np.array(selected)

            # set third_order_neighbors as padding number if the agent delete the second_order_neighbors
            selected_input = np.where(row_deleted_flag_2rd != 0, np.transpose(selected_input),
                                      self.padding_number_user_task)
            selected_input = np.transpose(selected_input)

            self.num_selected_3rd = self.batch_size * self.max_num_3rd - np.sum(
                selected_input == self.padding_number_user_task)

        return selected_input, row_deleted_index

    def cos_sim(self, target_ebd, source_ebd, step_index):
        '''
        calculate the cosine similarity between two vecotrs

        target_ebd b,e
        source_ebd b, n2, e

        '''
        vector1 = torch.from_numpy(target_ebd)  # [b, e]
        vector2 = torch.from_numpy(source_ebd[:, step_index, :])  # [b, e]
        output = torch.cosine_similarity(vector1, vector2, dim=1)
        output = np.reshape(output.numpy(), (-1, 1))

        return output

    def cal_element_wise(self, target_ebd, source_ebd, step_index):
        '''
        target_ebd b, e
        source_ebd b, n2, e
        '''
        vector1 = target_ebd
        vector2 = source_ebd[:, step_index, :]
        return np.multiply(vector1, vector2)  # [b, e]

    def update_state(self, action, state, step_index):
        self.action_matrix[:, step_index] = action
        self.state_matrix[:, step_index] = state

        self.num_selected = self.num_selected + action

        self.vector_current_sum1 = self.vector_current_sum1 + np.multiply(np.reshape(action, (-1, 1)),
                                                                          self.vector_current4)
        self.element_wise_current_sum1 = self.element_wise_current_sum1 + np.multiply(np.reshape(action, (-1, 1)),
                                                                                      self.element_wise_current1)
        self.element_wise_current_sum2 = self.element_wise_current_sum2 + np.multiply(np.reshape(action, (-1, 1)),
                                                                                      self.element_wise_current2)
        self.element_wise_current_sum3 = self.element_wise_current_sum3 + np.multiply(np.reshape(action, (-1, 1)),
                                                                                      self.element_wise_current3)

        num_selected_array = np.reshape(self.num_selected, (-1, 1))

        self.vector_current_mean1 = np.where(num_selected_array != 0, self.vector_current_sum1 / num_selected_array,
                                             self.vector_current_sum1)
        self.element_wise_current_mean1 = np.where(num_selected_array != 0,
                                                   self.element_wise_current_sum1 / num_selected_array,
                                                   self.element_wise_current_sum1)
        self.element_wise_current_mean2 = np.where(num_selected_array != 0,
                                                   self.element_wise_current_sum2 / num_selected_array,
                                                   self.element_wise_current_sum2)
        self.element_wise_current_mean3 = np.where(num_selected_array != 0,
                                                   self.element_wise_current_sum3 / num_selected_array,
                                                   self.element_wise_current_sum3)

    def update_state_3rd(self, action, state, step_index):
        self.action_matrix_3rd[:, step_index] = action
        self.state_matrix_3rd[:, step_index] = state
        self.num_selected_3rd = self.num_selected_3rd + action
        self.vector_current_sum1_3rd += np.multiply(np.reshape(action, (-1, 1)), self.vector_current4_3rd)
        self.element_wise_current_sum1_3rd += np.multiply(np.reshape(action, (-1, 1)), self.element_wise_current1_3rd)
        self.element_wise_current_sum2_3rd += np.multiply(np.reshape(action, (-1, 1)), self.element_wise_current2_3rd)
        self.element_wise_current_sum3_3rd += np.multiply(np.reshape(action, (-1, 1)), self.element_wise_current3_3rd)
        self.element_wise_current_sum4_3rd += np.multiply(np.reshape(action, (-1, 1)), self.element_wise_current4_3rd)
        self.element_wise_current_sum5_3rd += np.multiply(np.reshape(action, (-1, 1)), self.element_wise_current5_3rd)

        num_selected_array = np.reshape(self.num_selected, (-1, 1))
        self.vector_current_mean1_3rd = np.where(num_selected_array != 0,
                                                 self.vector_current_sum1_3rd / num_selected_array,
                                                 self.vector_current_sum1_3rd)
        self.element_wise_current_mean1_3rd = np.where(num_selected_array != 0,
                                                       self.element_wise_current_sum1_3rd / num_selected_array,
                                                       self.element_wise_current_sum1_3rd)
        self.element_wise_current_mean2_3rd = np.where(num_selected_array != 0,
                                                       self.element_wise_current_sum2_3rd / num_selected_array,
                                                       self.element_wise_current_sum2_3rd)
        self.element_wise_current_mean3_3rd = np.where(num_selected_array != 0,
                                                       self.element_wise_current_sum3_3rd / num_selected_array,
                                                       self.element_wise_current_sum3_3rd)
        self.element_wise_current_mean4_3rd = np.where(num_selected_array != 0,
                                                       self.element_wise_current_sum4_3rd / num_selected_array,
                                                       self.element_wise_current_sum4_3rd)
        self.element_wise_current_mean5_3rd = np.where(num_selected_array != 0,
                                                       self.element_wise_current_sum5_3rd / num_selected_array,
                                                       self.element_wise_current_sum5_3rd)

    def get_action_matrix(self):
        return self.action_matrix

    def get_state_matrix(self):
        return self.state_matrix

    def get_action_matrix_3rd(self):
        return self.action_matrix_3rd

    def get_state_matrix_3rd(self):
        return self.state_matrix_3rd

    def get_selected_second_order_neighbors_user_task(self, b_second_order_users):
        row_deleted_index = []

        # define row_deleted_flag, if each value equals to 0, it means delete, if each value equals to 1, it means maintain
        row_deleted_flag = np.ones((self.batch_size), dtype=np.int)

        selected_input = np.zeros((self.batch_size, self.max_num), dtype=np.int)
        for batch_index in range(self.batch_size):
            selected = []
            for neighbor_index in range(self.max_num):
                if self.action_matrix[batch_index, neighbor_index] == 1:
                    selected.append(b_second_order_users[batch_index, neighbor_index])
            # random select one course from the original enrolled courses if no course is selected by the agent, change the number of selected courses as 1 at the same time
            if len(selected) == 0:
                row_deleted_flag[batch_index] = 1

                row_deleted_index.append(batch_index)
                original_second_order_neghbors = list(set(b_second_order_users[batch_index]))
                if self.padding_number_user_task in original_second_order_neghbors:
                    original_second_order_neghbors.remove(self.padding_number_user_task)
                random_neighbor = np.random.choice(original_second_order_neghbors, 1)[0]
                selected.append(random_neighbor)
                self.num_selected[batch_index] = 1

            for neighbor_index in range(self.max_num - len(selected)):
                selected.append(self.padding_number_user_task)
            selected_input[batch_index, :] = np.array(selected)
        return selected_input, row_deleted_index, row_deleted_flag

    def get_selected_third_order_neighbors_user_task(self, b_third_order_items, row_deleted_flag_2rd):
        row_deleted_index = []
        selected_input = np.zeros((self.batch_size, self.max_num_3rd), dtype=np.int)
        for batch_index in range(self.batch_size):
            selected = []
            for neighbor_index in range(self.max_num_3rd):
                if self.action_matrix_3rd[batch_index, neighbor_index] == 1:
                    selected.append(b_third_order_items[batch_index, neighbor_index])
            if len(selected) == 0:
                row_deleted_index.append(batch_index)
                original_third_order_neighbors = list(set(b_third_order_items[batch_index]))
                if self.padding_number_item_task in original_third_order_neighbors:
                    original_third_order_neighbors.remove(self.padding_number_item_task)
                random_neighbor = np.random.choice(original_third_order_neighbors, 1)[0]
                selected.append(random_neighbor)

            for neighbor_index in range(self.max_num_3rd - len(selected)):
                selected.append(self.padding_number_item_task)
            selected_input[batch_index, :] = np.array(selected)

            # set third_order_neighbors as padding number if the agent delete the second_order_neighbors
            selected_input = np.where(row_deleted_flag_2rd != 0, np.transpose(selected_input),
                                      self.padding_number_item_task)
            selected_input = np.transpose(selected_input)

            self.num_selected_3rd = self.batch_size * self.max_num_3rd - np.sum(
                selected_input == self.padding_number_item_task)

        return selected_input, row_deleted_index


def get_action(prob, second_neighbor_column, padding_number):
    batch_size = prob.shape[0]
    random_number = np.random.rand(batch_size)
    return np.where((random_number < prob) & (second_neighbor_column != padding_number), np.ones(batch_size),
                    np.zeros(batch_size))


def decide_action(prob, second_neighbor_column, padding_number):
    batch_size = prob.shape[0]
    return np.where((prob >= 0.5) & (second_neighbor_column != padding_number), np.ones(batch_size),
                    np.zeros(batch_size))


def generate_batch_user_task(batch_index, model, sess, train_data, is_training, num_instances):
    predict_user_ebd = np.zeros(shape=(model.num_users, model.embedding_size))
    for index in batch_index:
        _, support_item = data.batch_gen_task(train_data, index, setting.batch_size, num_instances)
        feed_dict = {model.support_item: support_item,
                     model.training_phrase_user_task: is_training}
        batch_user_ebd = sess.run(model.final_support_encode_user_task, feed_dict)
        start_index = index * setting.batch_size
        end_index = min(start_index + setting.batch_size, num_instances)
        predict_user_ebd[start_index:end_index] = batch_user_ebd
    return predict_user_ebd


def generate_batch_item_task(batch_index, model, sess, train_data, is_training, num_instances):
    predict_item_ebd = np.zeros(shape=(model.num_items, model.embedding_size))
    for index in batch_index:
        all_item_id, support_user = data.batch_gen_task(train_data, index, setting.batch_size, num_instances)
        feed_dict = {model.support_user: support_user,
                     model.training_phrase_item_task: is_training}
        batch_item_ebd = sess.run(model.final_support_encode_item_task, feed_dict)

        item_id = list(all_item_id)
        predict_item_ebd[item_id] = batch_item_ebd
    return predict_item_ebd


def train_user_rl_task(name):
    env = environment()
    sample_times = setting.sample_times
    batch_size = setting.batch_size

    state_size = setting.state_size
    state_size_3rd = setting.third_order_state_size

    best_reward = -10000

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    g_rl = tf.Graph()
    sess2 = tf.Session(graph=g_rl, config=config)

    with g_rl.as_default():
        with sess2.as_default():
            model = GeneralGNN(name, sess2)
            rl_saver = tf.train.Saver(max_to_keep=3)
            rl_saver.restore(sess2, tf.train.get_checkpoint_state(
                os.path.dirname(setting.checkpoint_path_user_task + 'checkpoint')).model_checkpoint_path)

            g_rl.finalize()


            ClassData = data.Dataset(setting.oracle_training_file_user_task)

            for epoch in range(40): 
                train_batches = ClassData.get_positive_instances_user_task(epoch, 'train')
                num_batch_train = ClassData.oracle_num_users // setting.batch_size

                all_target_ori_ebd, all_target_aggre_ebd, all_target_second_aggre_ebd, all_first_neigh_ori_ebd, all_second_neigh_ori_ebd, all_third_neigh_ori_ebd, all_intra_user_2nd, all_intra_user_3rd, all_first_order_reward1, all_first_and_second_order_reward2, all_former_third_order_reward3 = model.generate_batch_state_ebd_user_task(
                    train_batches, ClassData)

                env.assign_batch_data(train_batches)
                for rl_epoch in range(1):
                    train_begin = time()
                    avg_total_reward = 0
                    avg_update_total_reward = 0
                    total_instances = 0
                    total_selected_second_neighbors = 0
                    total_selected_third_neighbors = 0
                    total_second_neighbors = 0
                    total_third_neighbors = 0

                    for batch_index in tqdm.tqdm(range(num_batch_train)):

                        b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item, b_intra_2nd_user, b_intra_3rd_user = env.parse_train_batches(
                            train_batches, batch_index)
                        b_target_ori_ebd, b_target_aggre_ebd, b_target_second_aggre_ebd, b_first_neigh_ori_ebd, b_second_neigh_ori_ebd, b_third_neigh_ori_ebd, b_intra_2nd_user, b_intra_3rd_user, b_prob1, b_prob2, b_prob3 = \
                            all_target_ori_ebd[
                                batch_index], \
                            all_target_aggre_ebd[
                                batch_index], \
                            all_target_second_aggre_ebd[
                                batch_index], \
                            all_first_neigh_ori_ebd[
                                batch_index], \
                            all_second_neigh_ori_ebd[
                                batch_index], all_third_neigh_ori_ebd[batch_index], \
                            all_intra_user_2nd[batch_index], all_intra_user_3rd[batch_index], all_first_order_reward1[
                                batch_index], all_first_and_second_order_reward2[
                                batch_index], all_former_third_order_reward3[batch_index]
                        original_reward_2rd = model.produce_original_reward_user_task(train_batches, num_batch_train,
                                                                                 ClassData)
                        original_reward_3rd = model.produce_original_reward_3rd_user_task(train_batches, num_batch_train,
                                                                                     ClassData)

                        max_num = b_second_neigh_ori_ebd.shape[1]

                        max_num_3rd = b_third_neigh_ori_ebd.shape[1]

                        batch_original_reward = np.reshape(b_prob3, (-1))

                        total_instances += b_target_ori_ebd.shape[0]

                        total_second_neighbors += np.sum(b_mask_num_second_order_user)
                        total_third_neighbors += np.sum(b_mask_num_third_order_item)


                        sampled_rewards = np.zeros((sample_times, batch_size), dtype=np.float32)
                        sampled_states = np.zeros((sample_times, batch_size, max_num, state_size), dtype=np.float32)
                        sampled_actions = np.zeros((sample_times, batch_size, max_num), dtype=np.float32)

                        sampled_rewards_3rd = np.zeros((sample_times, batch_size), dtype=np.float32)
                        sampled_states_3rd = np.zeros((sample_times, batch_size, max_num_3rd, state_size_3rd),
                                                      dtype=np.float32)
                        sampled_actions_3rd = np.zeros((sample_times, batch_size, max_num_3rd), dtype=np.float32)

                        avg_reward = np.zeros((batch_size), dtype=np.float32)
                        for sample_time in range(sample_times):
                            # 2rd
                            env.initilize_state_user_task(max_num, state_size, batch_original_reward,
                                                          original_reward_2rd)
                            for step_index in range(max_num):
                                state = env.get_state_user_task(b_prob1, b_prob2, b_target_ori_ebd,
                                                                b_target_aggre_ebd,
                                                                b_first_neigh_ori_ebd, b_second_neigh_ori_ebd,
                                                                step_index)

                              
                                prob = model.predict_second_order_target(state)

                                
                                b_second_order_users_array = np.array(b_second_order_users)
                            
                                action = get_action(prob, b_second_order_users_array[:, step_index],
                                                    env.padding_number_user_task)
                                env.update_state(action, state, step_index)
                            selected_input_2rd, row_deleted_index_2rd, row_deleted_flag_2rd = env.get_selected_second_order_neighbors_user_task(
                                b_second_order_users_array)

                            # 3rd
                            env.initilize_state_user_task_3rd(max_num_3rd, state_size_3rd)
                            for step_index in range(max_num_3rd):
                                state = env.get_state_user_task_3rd(b_target_ori_ebd, b_target_aggre_ebd,
                                                                    b_target_second_aggre_ebd,
                                                                    b_first_neigh_ori_ebd, b_second_neigh_ori_ebd,
                                                                    b_third_neigh_ori_ebd, step_index)

                                prob = model.predict_third_order_target(state)

                                b_3rd_order_users_array = np.array(b_third_order_items)
                                action = get_action(prob, b_3rd_order_users_array[:, step_index],
                                                    env.padding_number_user_task)
                                env.update_state_3rd(action, state, step_index)
                            selected_input_3rd, row_deleted_index_3rd = env.get_selected_third_order_neighbors_user_task(
                                b_3rd_order_users_array, row_deleted_flag_2rd)

                            reward, pearson = model.produce_reward_3rd_user_task(b_target_user, b_k_shot_item,
                                                                            selected_input_2rd, selected_input_3rd,
                                                                            b_oracle_user_ebd, b_intra_3rd_user)
                            # if selected neighbors is none in one row, we put the original reward to this situation
                            for del_idx in row_deleted_index_3rd:
                                reward[del_idx] = original_reward_3rd

                            reward = reward - env.batch_original_reward

                            avg_reward += reward
                            sampled_rewards[sample_time, :] = reward

                            sampled_actions[sample_time, :] = env.get_action_matrix()
                            sampled_states[sample_time, :] = env.get_state_matrix()

                            sampled_rewards_3rd[sample_time, :] = reward
                            sampled_actions_3rd[sample_time, :] = env.get_action_matrix_3rd()
                            sampled_states_3rd[sample_time, :] = env.get_state_matrix_3rd()

                        avg_reward = avg_reward / sample_times

                        second_order_gradbuffer = model.init_second_order_gradbuffer()
                        third_order_gradbuffer = model.init_third_order_gradbuffer()
                        for sample_time in range(sample_times):
                            reward_row = np.tile(
                                np.reshape(np.subtract(sampled_rewards[sample_time], avg_reward), (-1, 1)), max_num)
                            reward_row_3rd = np.tile(
                                np.reshape(np.subtract(sampled_rewards_3rd[sample_time], avg_reward), (-1, 1)),
                                max_num_3rd)

                            second_order_gradient = model.get_second_order_gradient(
                                np.reshape(sampled_states[sample_time], (-1, state_size)),
                                np.reshape(reward_row, (-1,)), np.reshape(sampled_actions[sample_time], (-1)))
                            model.train_second_order(second_order_gradbuffer, second_order_gradient)

                            third_order_gradient = model.get_third_order_gradient(
                                np.reshape(sampled_states_3rd[sample_time], (-1, state_size_3rd)),
                                np.reshape(reward_row_3rd, (-1)), np.reshape(sampled_actions_3rd[sample_time], (-1)))
                            model.train_third_order(third_order_gradbuffer, third_order_gradient)

                        # decision
                        # 2nd
                        env.initilize_state_user_task(max_num, state_size, batch_original_reward, original_reward_2rd)
                        for step_index in range(max_num):
                            state = env.get_state_user_task(b_prob1, b_prob2, b_target_ori_ebd,
                                                            b_target_aggre_ebd,
                                                            b_first_neigh_ori_ebd, b_second_neigh_ori_ebd,
                                                            step_index)
                            prob = model.predict_second_order_target(state)
                            b_second_order_users_array = np.array(b_second_order_users)
                            action = get_action(prob, b_second_order_users_array[:, step_index],
                                                env.padding_number_user_task)
                            env.update_state(action, state, step_index)
                        selected_input_2rd, row_deleted_index_2rd, row_deleted_flag_2rd = env.get_selected_second_order_neighbors_user_task(
                            b_second_order_users_array)

                        # 3rd
                        env.initilize_state_user_task_3rd(max_num_3rd, state_size_3rd)
                        for step_index in range(max_num_3rd):
                            state = env.get_state_user_task_3rd(b_target_ori_ebd, b_target_aggre_ebd,
                                                                b_target_second_aggre_ebd,
                                                                b_first_neigh_ori_ebd, b_second_neigh_ori_ebd,
                                                                b_third_neigh_ori_ebd, step_index)

                            prob = model.predict_third_order_target(state)
                            b_3rd_order_items_array = np.array(b_third_order_items)
                            action = get_action(prob, b_3rd_order_items_array[:, step_index],
                                                env.padding_number_user_task)
                            env.update_state_3rd(action, state, step_index)
                        selected_input_3rd, row_deleted_index_3rd = env.get_selected_third_order_neighbors_user_task(
                            b_3rd_order_items_array, row_deleted_flag_2rd)

                        reward, pearson = model.produce_reward_3rd_user_task(b_target_user, b_k_shot_item,
                                                                        selected_input_2rd, selected_input_3rd,
                                                                        b_oracle_user_ebd, b_intra_3rd_user)

                        # if selected neighbors is none in one row, we put the original reward to this situation
                        for del_idx in row_deleted_index_3rd:
                            reward[del_idx] = original_reward_3rd

                        reward = reward - env.batch_original_reward

                        avg_total_reward += np.sum(reward)
                        avg_update_total_reward += np.sum(reward)
                        total_selected_second_neighbors += np.sum(env.num_selected)
                        total_selected_third_neighbors += np.sum(env.num_selected_3rd)

                    # print total instances
                    avg_total_reward /= total_instances
                    avg_update_total_reward /= total_instances

                    if avg_update_total_reward > best_reward:
                        best_reward = avg_update_total_reward
                        rl_saver.save(sess2, save_path=setting.checkpoint_path_rl_user_task, global_step=epoch)

                    train_time = time() - train_begin
                    print(
                        'Epoch %d [%.1fs]:original_reward=%f,avg_reward=%f,update_reward=%f,best_reward=%f,total_second_order_neighbors=%d,selected_second_order_neighbors=%d, total_third_order_neighbors=%d, selected_third_order_neighbors=%d'
                        % (
                            epoch, train_time, env.original_reward, avg_total_reward, avg_update_total_reward,
                            best_reward,
                            total_second_neighbors, total_selected_second_neighbors, total_third_neighbors,
                            total_selected_third_neighbors))

                    # update the rlmodel apply gradient#
                    model.update_target_second_order_network()
                    model.update_target_third_order_network()


            print("Evaluate pre-trained aggregator based on original test instances...")
            avg_cosine_similarity = model.evaluate_3rd_user_task()
            print('epoch', -1, avg_cosine_similarity)

def revise_rl_user_task(name):
    print("Evaluate pre-trained aggregator based on revised test intances")
    print('restore agent...')
    env = environment()

    state_size = setting.state_size
    state_size_3rd = setting.third_order_state_size
    total_reward, total_pearson = 0.0, 0.0

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    g_rl_revise = tf.Graph()
    sess3 = tf.Session(graph=g_rl_revise, config=config)
    with g_rl_revise.as_default():
        with sess3.as_default():
            model = GeneralGNN(name, sess3)
            rl_saver = tf.train.Saver()
            rl_saver.restore(sess3, tf.train.get_checkpoint_state(
                os.path.dirname(setting.checkpoint_path_rl_user_task + 'checkpoint')).model_checkpoint_path)

            data_valid = data.Dataset(setting.oracle_valid_file_user_task)
            valid_batches = data_valid.get_positive_instances_user_task(0, 'valid')
            num_batch_valid = data_valid.oracle_num_users // setting.batch_size
            valid_batch_index = range(num_batch_valid)

            all_target_ori_ebd, all_target_aggre_ebd, all_target_second_aggre_ebd, all_first_neigh_ori_ebd, all_second_neigh_ori_ebd, all_third_neigh_ori_ebd, all_intra_2nd_user, all_intra_3rd_user, all_first_order_reward1, all_first_and_second_order_reward2, all_former_third_order_reward3 = model.generate_batch_state_ebd_user_task(
                valid_batches, data_valid)

            for batch_index in tqdm.tqdm(valid_batch_index):
                b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item, b_intra_2nd_user, b_intra_3rd_user = data_valid.batch_gen_3rd_user_task(
                    valid_batches, batch_index)

                b_target_ori_ebd, b_target_aggre_ebd, b_target_second_aggre_ebd, b_first_neigh_ori_ebd, b_second_neigh_ori_ebd, b_third_neigh_ori_ebd, b_prob1, b_prob2, b_prob3 = \
                    all_target_ori_ebd[
                        batch_index], \
                    all_target_aggre_ebd[
                        batch_index], \
                    all_target_second_aggre_ebd[
                        batch_index], \
                    all_first_neigh_ori_ebd[
                        batch_index], \
                    all_second_neigh_ori_ebd[
                        batch_index], all_third_neigh_ori_ebd[batch_index], all_first_order_reward1[
                        batch_index], all_first_and_second_order_reward2[
                        batch_index], all_former_third_order_reward3[batch_index]

                max_num = b_second_neigh_ori_ebd.shape[1]
                max_num_3rd = b_third_neigh_ori_ebd.shape[1]

                original_reward_2rd = model.produce_original_reward_user_task(valid_batches, num_batch_valid, data_valid)
                original_reward_3rd = model.produce_original_reward_3rd_user_task(valid_batches, num_batch_valid, data_valid)

                batch_original_reward = np.reshape(b_prob3, (-1))

                env.initilize_state_user_task(max_num, state_size, batch_original_reward, original_reward_2rd)
                for step_index in range(max_num):
                    state = env.get_state_user_task(b_prob1, b_prob2, b_target_ori_ebd,
                                                    b_target_aggre_ebd,
                                                    b_first_neigh_ori_ebd, b_second_neigh_ori_ebd,
                                                    step_index)

                    prob = model.predict_second_order_target(state)

                    b_second_order_users_array = np.array(b_second_order_users)
                    action = get_action(prob, b_second_order_users_array[:, step_index],
                                        env.padding_number_user_task)
                    env.update_state(action, state, step_index)
                selected_input_2rd, row_deleted_index_2rd, row_deleted_flag_2rd = env.get_selected_second_order_neighbors_user_task(
                    b_second_order_users_array)

                # 3rd
                env.initilize_state_user_task_3rd(max_num_3rd, state_size_3rd)
                for step_index in range(max_num_3rd):
                    state = env.get_state_user_task_3rd(b_target_ori_ebd, b_target_aggre_ebd,
                                                        b_target_second_aggre_ebd,
                                                        b_first_neigh_ori_ebd, b_second_neigh_ori_ebd,
                                                        b_third_neigh_ori_ebd, step_index)
                    prob = model.predict_third_order_target(state)
                   
                    b_3rd_order_items_array = np.array(b_third_order_items)
                    action = get_action(prob, b_3rd_order_items_array[:, step_index],
                                        env.padding_number_user_task)
                    env.update_state_3rd(action, state, step_index)
                selected_input_3rd, row_deleted_index_3rd = env.get_selected_third_order_neighbors_user_task(
                    b_3rd_order_items_array, row_deleted_flag_2rd)

                reward, pearson = model.produce_reward_3rd_item_task(b_target_user, b_k_shot_item, selected_input_2rd,
                                                                selected_input_3rd,
                                                                b_oracle_user_ebd, b_intra_3rd_user)
                # if selected neighbors is none in one row, we put the original reward to this situation
                for del_idx in row_deleted_index_3rd:
                    reward[del_idx] = original_reward_3rd

                reward = np.mean(reward)

                total_reward += reward

                total_pearson += pearson

            avg_reward = total_reward / num_batch_valid
            avg_pearson = total_pearson / num_batch_valid
            print('epoch', 0, avg_reward, avg_pearson)


def train_item_rl_task(name):
    env = environment()
    sample_times = setting.sample_times
    batch_size = setting.batch_size

    state_size = setting.state_size
    state_size_3rd = setting.third_order_state_size

    best_reward = -10000

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    g_rl = tf.Graph()
    sess2 = tf.Session(graph=g_rl, config=config)

    with g_rl.as_default():
        with sess2.as_default():
            model = GeneralGNN(name, sess2)
            rl_saver = tf.train.Saver(max_to_keep=3)
            rl_saver.restore(sess2, tf.train.get_checkpoint_state(
                os.path.dirname(setting.checkpoint_path_item_task + 'checkpoint')).model_checkpoint_path)
            g_rl.finalize()

            ClassData = data.Dataset(setting.oracle_training_file_item_task)

            for epoch in range(40):
                train_batches = ClassData.get_positive_instances_item_task(epoch, 'train')
                num_batch_train = ClassData.oracle_num_items // setting.batch_size

                all_target_ori_ebd, all_target_aggre_ebd, all_target_second_aggre_ebd, all_first_neigh_ori_ebd, all_second_neigh_ori_ebd, all_third_neigh_ori_ebd, all_intra_2nd_ori_ebd, all_intra_3rd_ori_ebd, all_first_order_reward1, all_first_and_second_order_reward2, all_former_third_order_reward3 = model.generate_batch_state_ebd_item_task(
                    train_batches, ClassData)

                env.assign_batch_data(train_batches)
                for rl_epoch in range(1):
                    train_begin = time()
                    avg_total_reward = 0
                    avg_update_total_reward = 0
                    total_instances = 0
                    total_selected_second_neighbors = 0
                    total_selected_third_neighbors = 0
                    total_second_neighbors = 0
                    total_third_neighbors = 0

                    for batch_index in tqdm.tqdm(range(num_batch_train)):

                        b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user, b_intra_2nd_item, b_intra_3rd_item = env.parse_train_batches(
                            train_batches, batch_index)
                        b_target_ori_ebd, b_target_aggre_ebd, b_target_second_aggre_ebd, b_first_neigh_ori_ebd, b_second_neigh_ori_ebd, batch_third_neigh_ori_ebd, batch_intra_2nd_item, batch_intra_3rd_item, b_prob1, b_prob2, b_prob3 = \
                            all_target_ori_ebd[
                                batch_index], \
                            all_target_aggre_ebd[
                                batch_index], \
                            all_target_second_aggre_ebd[batch_index], \
                            all_first_neigh_ori_ebd[
                                batch_index], \
                            all_second_neigh_ori_ebd[
                                batch_index], all_third_neigh_ori_ebd[batch_index], all_intra_2nd_ori_ebd[batch_index], \
                            all_intra_3rd_ori_ebd[batch_index], all_first_order_reward1[
                                batch_index], all_first_and_second_order_reward2[
                                batch_index], all_former_third_order_reward3[batch_index]
                        original_reward_2rd = model.produce_original_reward_item_task(train_batches, num_batch_train,
                                                                                 ClassData)
                        original_reward_3rd = model.produce_original_reward_3rd_item_task(train_batches, num_batch_train,
                                                                                     ClassData)

                        max_num = b_second_neigh_ori_ebd.shape[1]

                        max_num_3rd = batch_third_neigh_ori_ebd.shape[1]

                        batch_original_reward = np.reshape(b_prob3, (-1))

                        total_instances += b_target_ori_ebd.shape[0]

                        total_second_neighbors += np.sum(b_mask_num_second_order_item)
                        total_third_neighbors += np.sum(b_mask_num_third_order_user)


                        sampled_rewards = np.zeros((sample_times, batch_size), dtype=np.float32)
                        sampled_states = np.zeros((sample_times, batch_size, max_num, state_size), dtype=np.float32)
                        sampled_actions = np.zeros((sample_times, batch_size, max_num), dtype=np.float32)

                        sampled_rewards_3rd = np.zeros((sample_times, batch_size), dtype=np.float32)
                        sampled_states_3rd = np.zeros((sample_times, batch_size, max_num_3rd, state_size_3rd),
                                                      dtype=np.float32)
                        sampled_actions_3rd = np.zeros((sample_times, batch_size, max_num_3rd), dtype=np.float32)

                        avg_reward = np.zeros((batch_size), dtype=np.float32)
                        for sample_time in range(sample_times):
                            # 2rd
                            env.initilize_state_item_task(max_num, state_size, batch_original_reward,
                                                          original_reward_3rd)
                            for step_index in range(max_num):
                                state = env.get_state_item_task(b_prob1, b_prob2, b_target_ori_ebd,
                                                                b_target_aggre_ebd,
                                                                b_first_neigh_ori_ebd, b_second_neigh_ori_ebd,
                                                                step_index)
                                prob = model.predict_second_order_target(state)
                                b_second_order_items_array = np.array(b_second_order_items)
                                action = get_action(prob, b_second_order_items_array[:, step_index],
                                                    env.padding_number_item_task)
                                env.update_state(action, state, step_index)
                            selected_input_2rd, row_deleted_index_2rd, row_deleted_flag_2rd = env.get_selected_second_order_neighbors_item_task(
                                b_second_order_items_array)

                            # 3rd
                            env.initilize_state_item_task_3rd(max_num_3rd, state_size_3rd)
                            for step_index in range(max_num_3rd):
                                state = env.get_state_item_task_3rd(b_target_ori_ebd, b_target_aggre_ebd,
                                                                    b_target_second_aggre_ebd,
                                                                    b_first_neigh_ori_ebd, b_second_neigh_ori_ebd,
                                                                    batch_third_neigh_ori_ebd, step_index)
                                prob = model.predict_third_order_target(state)
                                b_3rd_order_users_array = np.array(b_third_order_users)
                                action = get_action(prob, b_3rd_order_users_array[:, step_index],
                                                    env.padding_number_item_task)
                                env.update_state_3rd(action, state_size_3rd, step_index)
                            selected_input_3rd, row_deleted_index_3rd = env.get_selected_third_order_neighbors_item_task(
                                b_3rd_order_users_array, row_deleted_flag_2rd)

                            reward, pearson = model.produce_reward_item_task(b_target_item, b_k_shot_user,
                                                                        selected_input_2rd,
                                                                        b_oracle_item_ebd, b_intra_2nd_item)
                            # if selected neighbors is none in one row, we put the original reward to this situation
                            for del_idx in row_deleted_index_3rd:
                                reward[del_idx] = original_reward_3rd

                            reward = reward - env.batch_original_reward

                            avg_reward += reward

                            sampled_rewards[sample_time, :] = reward
                            sampled_actions[sample_time, :] = env.get_action_matrix()
                            sampled_states[sample_time, :] = env.get_state_matrix()

                            sampled_rewards_3rd[sample_time, :] = reward
                            sampled_actions_3rd[sample_time, :] = env.get_action_matrix_3rd()
                            sampled_states_3rd[sample_time, :] = env.get_state_matrix_3rd()

                        avg_reward = avg_reward / sample_times

                        second_order_gradbuffer = model.init_second_order_gradbuffer()
                        third_order_gradbuffer = model.init_third_order_gradbuffer()
                        # update agent, using object function
                        for sample_time in range(sample_times):
                            reward_row = np.tile(
                                np.reshape(np.subtract(sampled_rewards[sample_time], avg_reward), (-1, 1)), max_num)
                            reward_row_3rd = np.tile(
                                np.reshape(np.subtract(sampled_rewards_3rd[sample_time], avg_reward), (-1, 1)),
                                max_num_3rd)

                            second_order_gradient = model.get_second_order_gradient(
                                np.reshape(sampled_states[sample_time], (-1, state_size)),
                                np.reshape(reward_row, (-1,)), np.reshape(sampled_actions[sample_time], (-1)))
                            model.train_second_order(second_order_gradbuffer, second_order_gradient)

                            third_order_gradient = model.get_third_order_gradient(
                                np.reshape(sampled_states_3rd[sample_time], (-1, state_size_3rd)),
                                np.reshape(reward_row_3rd, (-1)), np.reshape(sampled_actions_3rd[sample_time], (-1)))
                            model.train_third_order(third_order_gradbuffer, third_order_gradient)

                        # decision
                        # 2rd
                        env.initilize_state_item_task(max_num, state_size, batch_original_reward, original_reward_3rd)
                        for step_index in range(max_num):
                            state = env.get_state_item_task(b_prob1, b_prob2, b_target_ori_ebd,
                                                            b_target_aggre_ebd,
                                                            b_first_neigh_ori_ebd, b_second_neigh_ori_ebd,
                                                            step_index)
                            prob = model.predict_second_order_target(state)
                            b_second_order_users_array = np.array(b_second_order_items)
                            action = decide_action(prob, b_second_order_users_array[:, step_index],
                                                   env.padding_number_item_task)
                            env.update_state(action, state, step_index)
                        selected_input_2rd, row_deleted_index_2rd, row_deleted_flag_2rd = env.get_selected_second_order_neighbors_item_task(
                            b_second_order_users_array)

                        # 3rd
                        env.initilize_state_item_task_3rd(max_num_3rd, state_size_3rd)
                        for step_index in range(max_num_3rd):
                            state = env.get_state_item_task_3rd(b_target_ori_ebd, b_target_aggre_ebd,
                                                                b_target_second_aggre_ebd,
                                                                b_first_neigh_ori_ebd, b_second_neigh_ori_ebd,
                                                                batch_third_neigh_ori_ebd, step_index)
                            prob = model.predict_third_order_target(state)
                            b_3rd_order_users_array = np.array(b_third_order_users)
                            action = get_action(prob, b_3rd_order_users_array[:, step_index],
                                                env.padding_number_item_task)
                            env.update_state_3rd(action, state_size_3rd, step_index)
                        selected_input_3rd, row_deleted_index_3rd = env.get_selected_third_order_neighbors_item_task(
                            b_3rd_order_users_array, row_deleted_flag_2rd)

                        reward, pearson = model.produce_reward_3rd_item_task(b_target_item, b_k_shot_user,
                                                                        selected_input_2rd,
                                                                        selected_input_3rd,
                                                                        b_oracle_item_ebd, b_intra_3rd_item)
                        for del_idx in row_deleted_index_3rd:
                            reward[del_idx] = original_reward_3rd
                        reward = reward - env.batch_original_reward

                        avg_total_reward += np.sum(reward)
                        avg_update_total_reward += np.sum(reward)
                        total_selected_second_neighbors += np.sum(env.num_selected)
                        total_selected_third_neighbors += np.sum(env.num_selected_3rd)

                    # print total instances
                    avg_total_reward /= total_instances
                    avg_update_total_reward /= total_instances

                    if avg_update_total_reward > best_reward:
                        best_reward = avg_update_total_reward
                        rl_saver.save(sess2, save_path=setting.checkpoint_path_rl_item_task, global_step=epoch)

                    train_time = time() - train_begin
                    print(
                        'Epoch %d [%.1fs]:original_reward=%f,avg_reward=%f,update_reward=%f,best_reward=%f,total_second_order_neighbors=%d,selected_second_order_neighbors=%d, total_third_order_neighbors=%d, selected_third_order_neighbors=%d'
                        % (
                            epoch, train_time, env.original_reward, avg_total_reward, avg_update_total_reward,
                            best_reward,
                            total_second_neighbors, total_selected_second_neighbors, total_third_neighbors,
                            total_selected_third_neighbors))

                    # update the rlmodel apply gradient#
                    model.update_target_second_order_network()
                    model.update_target_third_order_network()

            print("Evaluate pre-trained aggregator based on original test instances...")
            avg_cosine_similarity = model.evaluate_3rd_item_task()
            print('epoch', -1, avg_cosine_similarity)

def revise_rl_item_task(name):
    print("Evaluate pre-trained aggregator based on revised test intances")
    print('restore agent...')
    env = environment()

    state_size = setting.state_size
    state_size_3rd = setting.state_size_3rd
    total_reward, total_pearson = 0.0, 0.0

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    g_rl_revise = tf.Graph()
    sess3 = tf.Session(graph=g_rl_revise, config=config)
    with g_rl_revise.as_default():
        with sess3.as_default():
            model = GeneralGNN(name, sess3)
            rl_saver = tf.train.Saver(max_to_keep=3)
            rl_saver.restore(sess3, tf.train.get_checkpoint_state(
                os.path.dirname(setting.checkpoint_path_rl_item_task + 'checkpoint')).model_checkpoint_path)

            data_valid = data.Dataset(setting.oracle_valid_file_item_task)
            valid_batches = data_valid.get_positive_instances_item_task(0, 'valid')
            num_batch_valid = data_valid.oracle_num_items // setting.batch_size
            valid_batch_index = range(num_batch_valid)

            all_target_ori_ebd, all_target_aggre_ebd, all_target_second_aggre_ebd, all_first_neigh_ori_ebd, all_second_neigh_ori_ebd, all_third_neigh_ori_ebd, all_first_order_reward1, all_intra_2nd_ori_ebd, all_intra_3rd_ori_ebd, all_first_and_second_order_reward2, all_former_third_order_reward3 = model.generate_batch_state_ebd_item_task(
                valid_batches, data_valid)

            for batch_index in tqdm.tqdm(valid_batch_index):

                b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user, b_intra_2nd_item, b_intra_3rd_item = env.parse_train_batches(
                    valid_batches, batch_index)

                b_target_ori_ebd, b_target_aggre_ebd, b_target_second_aggre_ebd, b_first_neigh_ori_ebd, b_second_neigh_ori_ebd, b_third_neigh_ori_ebd, b_intra_2nd_item, b_intra_3rd_item, b_prob1, b_prob2, b_prob3 = \
                    all_target_ori_ebd[
                        batch_index], \
                    all_target_aggre_ebd[
                        batch_index], \
                    all_target_second_aggre_ebd[
                        batch_index], \
                    all_first_neigh_ori_ebd[
                        batch_index], \
                    all_second_neigh_ori_ebd[
                        batch_index], all_third_neigh_ori_ebd[batch_index], all_intra_2nd_ori_ebd[batch_index], \
                    all_intra_3rd_ori_ebd[batch_index], \
                    all_first_order_reward1[batch_index], all_first_and_second_order_reward2[batch_index], \
                    all_former_third_order_reward3[batch_index]

                max_num = b_second_neigh_ori_ebd.shape[1]
                max_num_3rd = b_third_neigh_ori_ebd.shape[1]

                original_reward_2rd = model.produce_original_reward_item_task(valid_batches, num_batch_valid, data_valid)
                original_reward_3rd = model.produce_original_reward_3rd_item_task(valid_batches, num_batch_valid, data_valid)

                batch_original_reward = np.reshape(b_prob3, (-1))

                env.initilize_state_item_task(max_num, state_size, batch_original_reward, original_reward_2rd)
                for step_index in range(max_num):
                    state = env.get_state_item_task(b_prob1, b_prob2, b_target_ori_ebd,
                                                    b_target_aggre_ebd,
                                                    b_first_neigh_ori_ebd, b_second_neigh_ori_ebd,
                                                    step_index)

                    prob = model.predict_second_order_target(state)

                    b_second_order_item_array = np.array(b_second_order_items)
                    action = get_action(prob, b_second_order_item_array[:, step_index],
                                        env.padding_number_user_task)
                    env.update_state(action, state, step_index)
                selected_input_2rd, row_deleted_index_2rd, row_deleted_flag_2rd = env.get_selected_second_order_neighbors_item_task(
                    b_second_order_item_array)

                # 3rd
                env.initilize_state_item_task_3rd(max_num_3rd, state_size_3rd)
                for step_index in range(max_num_3rd):
                    state = env.get_state_item_task_3rd(b_target_ori_ebd, b_target_aggre_ebd,
                                                        b_target_second_aggre_ebd,
                                                        b_first_neigh_ori_ebd, b_second_neigh_ori_ebd,
                                                        b_third_neigh_ori_ebd, step_index)

                    prob = model.predict_third_order_target(state)

                    b_3rd_order_users_array = np.array(b_third_order_users)
                    action = get_action(prob, b_3rd_order_users_array[:, step_index],
                                        env.padding_number_user_task)
                    env.update_state_3rd(action, state, step_index)
                selected_input_3rd, row_deleted_index_3rd = env.get_selected_third_order_neighbors_item_task(
                    b_3rd_order_users_array, row_deleted_flag_2rd)

                reward, pearson = model.produce_reward_3rd_item_task(b_target_item, b_k_shot_user, selected_input_2rd,
                                                                selected_input_3rd,
                                                                b_oracle_item_ebd, b_intra_3rd_item)
                # if selected neighbors is none in one row, we put the original reward to this situation
                for del_idx in row_deleted_index_3rd:
                    reward[del_idx] = original_reward_3rd

                reward = np.mean(reward)

                total_reward += reward

                total_pearson += pearson

            avg_reward = total_reward / num_batch_valid
            avg_pearson = total_pearson / num_batch_valid
            print('epoch', 0, avg_reward, avg_pearson)

