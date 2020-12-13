import numpy as np
import setting
import scipy.sparse as sp
import random
from downstream_trainer import *


class DataLoader(object):
    def __init__(self, path1, path2):
        self.num_users = setting.num_users
        self.num_items = setting.num_items

        self.oracle_num_items = setting.num_items
        self.oracle_num_users = setting.num_users

        self.padding_number_items = self.num_items
        self.padding_number_users = self.num_users

        self.oracle_user_ebd = np.load(setting.oracle_user_ebd_path)
        self.oracle_item_ebd = np.load(setting.oracle_item_ebd_path)

        self.train = self._load_rating_file_as_matrix(path1, path2)
        self._load_original_rating_file_as_list(path1, path2)

    def _load_rating_file_as_matrix(self, path1, path2):
        # Construct matrix
        mat = sp.dok_matrix((setting.num_users, setting.num_items), dtype=np.int32)
        with open(path1, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(",")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()

        with open(path2, "r") as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(',')
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    def _load_original_rating_file_as_list(self, file1, file2):
        self.neighbor_dict_user_list, self.neighbor_dict_item_list = {}, {}
        with open(file1, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(',')
                user, item = int(arr[0]), int(arr[1])
                if user not in self.neighbor_dict_user_list:
                    self.neighbor_dict_user_list[user] = []
                    self.neighbor_dict_user_list[user].append(item)
                else:
                    self.neighbor_dict_user_list[user].append(item)

                if item not in self.neighbor_dict_item_list:
                    self.neighbor_dict_item_list[item] = []
                    self.neighbor_dict_item_list[item].append(user)
                else:
                    self.neighbor_dict_item_list[item].append(user)
                line = f.readline()

        with open(file2, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(',')
                user, item = int(arr[0]), int(arr[1])
                if user not in self.neighbor_dict_user_list:
                    self.neighbor_dict_user_list[user] = []
                    self.neighbor_dict_user_list[user].append(item)
                else:
                    self.neighbor_dict_user_list[user].append(item)

                if item not in self.neighbor_dict_item_list:
                    self.neighbor_dict_item_list[item] = []
                    self.neighbor_dict_item_list[item].append(user)
                else:
                    self.neighbor_dict_item_list[item].append(user)
                line = f.readline()

        # padding, if the number of user and item is not in range(num_items) and range(num_users)
        for user in range(self.num_users):
            if user not in self.neighbor_dict_user_list.keys():
                self.neighbor_dict_user_list[user] = []
                self.neighbor_dict_user_list[user].append(0)  # padding

        for item in range(self.num_items):
            if item not in self.neighbor_dict_item_list.keys():
                self.neighbor_dict_item_list[item] = []
                self.neighbor_dict_item_list[item].append(0)

    def get_train_instances(self, epoch):
        np.random.seed(epoch)
        test_user_ground_truth = {}
        train_dict = {} # all train user-item pairs
        training_dict = {} # sampled user-item pairs
        with open(setting.downstream_query_file, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(',')
                user, item = int(arr[0]), int(arr[1])
                if user not in test_user_ground_truth:
                    test_user_ground_truth[user] = []
                    test_user_ground_truth[user].append(item)
                else:
                    test_user_ground_truth[user].append(item)

                line = f.readline()
        with open(setting.downstream_meta_train_file, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(',')
                user, item = int(arr[0]), int(arr[1])
                if user not in test_user_ground_truth:
                    test_user_ground_truth[user] = []
                    test_user_ground_truth[user].append(item)
                else:
                    test_user_ground_truth[user].append(item)

                if user not in train_dict:
                    train_dict[user] = []
                    train_dict[user].append(item)
                else:
                    train_dict[user].append(item)
 
                line = f.readline()

        for user in train_dict.keys():
            training_dict[user] = []
            temp_list = train_dict[user]
            random.shuffle(temp_list)
            if len(temp_list) >6:
                temp_list = temp_list[:6]
            for item in temp_list:
                training_dict[user].append(item)



        user_input, pos_item_input, neg_item_input = [], [], []
        u_1st_item, u_2nd_user, u_3rd_item, u_oracle_user_ebd, u_mask_num_2nd_user, u_mask_num_3rd_item = [], [], [], [], [], []
        ipos_1st_user, ipos_2nd_item, ipos_3rd_user, ipos_oracle_item_ebd, ipos_mask_num_2nd_item, ipos_mask_num_3rd_user = [], [], [], [], [], []
        ineg_1st_user, ineg_2nd_item, ineg_3rd_user, ineg_oracle_item_ebd, ineg_mask_num_2nd_item, ineg_mask_num_3rd_user = [], [], [], [], [], []

        num_users, num_items = setting.num_users, setting.num_items
        # for (u, i) in train.keys():
        for u in training_dict.keys():
            for i in training_dict[u]:
                user_input.append(u)
                pos_item_input.append(i)
                j = np.random.randint(num_items)
                while (u, j) in self.train.keys() or j in test_user_ground_truth[u]:
                    j = np.random.randint(num_items)
                neg_item_input.append(j)

                # user neighbor
                per_1st_i, per_2nd_u, per_3rd_i, per_oracle_u_ebd, per_mask_num_2nd_u, per_mask_num_3rd_i = self._get_user_neighbors(u)

                u_1st_item.append(per_1st_i)
                u_2nd_user.append(per_2nd_u)
                u_3rd_item.append(per_3rd_i)
                u_oracle_user_ebd.append(per_oracle_u_ebd)
                u_mask_num_2nd_user.append(per_mask_num_2nd_u)
                u_mask_num_3rd_item.append(per_mask_num_3rd_i)

                # positive item neighbor
                per_1st_upos, per_2nd_ipos, per_3rd_upos, per_oracle_ipos_ebd, per_mask_num_2nd_ipos, per_mask_num_3rd_upos = self._get_item_neighbors(i)

                ipos_1st_user.append(per_1st_upos)
                ipos_2nd_item.append(per_2nd_ipos)
                ipos_3rd_user.append(per_3rd_upos)
                ipos_oracle_item_ebd.append(per_oracle_ipos_ebd)
                ipos_mask_num_2nd_item.append(per_mask_num_2nd_ipos)
                ipos_mask_num_3rd_user.append(per_mask_num_3rd_upos)

                # negative item neighbor
                per_1st_uneg, per_2nd_ineg, per_3rd_uneg, per_oracle_ineg_ebd, per_mask_num_2nd_ineg, per_mask_num_3rd_uneg = self._get_item_neighbors(j)

                ineg_1st_user.append(per_1st_uneg)
                ineg_2nd_item.append(per_2nd_ineg)
                ineg_3rd_user.append(per_3rd_uneg)
                ineg_oracle_item_ebd.append(per_oracle_ineg_ebd)
                ineg_mask_num_2nd_item.append(per_mask_num_2nd_ineg)
                ineg_mask_num_3rd_user.append(per_mask_num_3rd_uneg)

        user_input = np.array(user_input)
        u_1st_item = np.array(u_1st_item)
        u_2nd_user = np.array(u_2nd_user)
        u_3rd_item = np.array(u_3rd_item)
        u_oracle_user_ebd = np.array(u_oracle_user_ebd)
        u_mask_num_2nd_user = np.array(u_mask_num_2nd_user)
        u_mask_num_3rd_item = np.array(u_mask_num_3rd_item)
        pos_item_input = np.array(pos_item_input)
        ipos_1st_user = np.array(ipos_1st_user)
        ipos_2nd_item = np.array(ipos_2nd_item)
        ipos_3rd_user = np.array(ipos_3rd_user)
        ipos_oracle_item_ebd = np.array(ipos_oracle_item_ebd)
        ipos_mask_num_2nd_item = np.array(ipos_mask_num_2nd_item)
        ipos_mask_num_3rd_user = np.array(ipos_mask_num_3rd_user)
        neg_item_input = np.array(neg_item_input)
        ineg_1st_user = np.array(ineg_1st_user)
        ineg_2nd_item = np.array(ineg_2nd_item)
        ineg_3rd_user = np.array(ineg_3rd_user)
        ineg_oracle_item_ebd = np.array(ineg_oracle_item_ebd)
        ineg_mask_num_2nd_item = np.array(ineg_mask_num_2nd_item)
        ineg_mask_num_3rd_user = np.array(ineg_mask_num_3rd_user)

        return user_input, u_1st_item, u_2nd_user, u_3rd_item, u_oracle_user_ebd, u_mask_num_2nd_user, u_mask_num_3rd_item, pos_item_input, ipos_1st_user, ipos_2nd_item, ipos_3rd_user, ipos_oracle_item_ebd, ipos_mask_num_2nd_item, ipos_mask_num_3rd_user, neg_item_input, ineg_1st_user, ineg_2nd_item, ineg_3rd_user, ineg_oracle_item_ebd, ineg_mask_num_2nd_item, ineg_mask_num_3rd_user

    def _get_user_neighbors(self, user):
        per_oracle_user_ebd = self.oracle_user_ebd[user]
        if len(self.neighbor_dict_user_list[user]) > setting.kshot_num:
            per_1st_item = np.random.choice(self.neighbor_dict_user_list[user], setting.kshot_num,
                                            replace=False)
        else:
            per_1st_item = np.random.choice(self.neighbor_dict_user_list[user], setting.kshot_num,
                                            replace=True)
        current_second_order_user = []
        for each_kshot_item in per_1st_item:

            candidate_second_order_user = self.neighbor_dict_item_list[each_kshot_item]

            if user in candidate_second_order_user:
                candidate_second_order_user.remove(user)

            if len(candidate_second_order_user) == 0:
                candidate_second_order_user.append(user)

            if len(candidate_second_order_user) < setting.kshot_second_num:
                per_2nd_user = list(
                    np.random.choice(candidate_second_order_user, setting.kshot_second_num, replace=True))
            else:
                per_2nd_user = list(
                    np.random.choice(candidate_second_order_user, setting.kshot_second_num, replace=False))

            current_second_order_user += per_2nd_user
        per_2nd_user = list(set(current_second_order_user))
        per_mask_num_2nd_user = len(per_2nd_user)
        pad_per_2nd_user = per_2nd_user + [self.padding_number_users] * (
                setting.kshot_num * setting.kshot_second_num - len(per_2nd_user))

        current_third_order_item = []
        for each_kshot_user in per_2nd_user:
            candidate_third_order_item = self.neighbor_dict_user_list[each_kshot_user]

            if len(candidate_third_order_item) < setting.kshot_third_num:
                per_3rd_item = list(
                    np.random.choice(candidate_third_order_item, setting.kshot_third_num, replace=True))
            else:
                per_3rd_item = list(
                    np.random.choice(candidate_third_order_item, setting.kshot_third_num, replace=False))

            current_third_order_item += per_3rd_item
        per_3rd_item = list(set(current_third_order_item))
        per_mask_num_3rd_item = len(per_3rd_item)
        pad_per_3rd_item = per_3rd_item + [self.padding_number_items] * (
                setting.kshot_num * setting.kshot_second_num * setting.kshot_third_num - len(per_3rd_item))



        return per_1st_item, pad_per_2nd_user, pad_per_3rd_item, per_oracle_user_ebd, per_mask_num_2nd_user, per_mask_num_3rd_item

    def _get_item_neighbors(self, item):
        per_oracle_item_ebd = self.oracle_item_ebd[item]

        if len(self.neighbor_dict_item_list[item]) < setting.kshot_num:
            per_1st_user = np.random.choice(self.neighbor_dict_item_list[item], setting.kshot_num,
                                            replace=True)
        else:
            per_1st_user = np.random.choice(self.neighbor_dict_item_list[item], setting.kshot_num,
                                            replace=False)

        current_second_order_item = []
        for each_kshot_user in per_1st_user:
            candidate_second_order_item = self.neighbor_dict_user_list[each_kshot_user]

            if item in candidate_second_order_item:
                candidate_second_order_item.remove(item)
            if len(candidate_second_order_item) == 0:
                candidate_second_order_item.append(item)

            if len(candidate_second_order_item) < setting.kshot_second_num:
                per_2nd_item = list(
                    np.random.choice(candidate_second_order_item, setting.kshot_second_num, replace=True))
            else:
                per_2nd_item = list(
                    np.random.choice(candidate_second_order_item, setting.kshot_second_num, replace=False))

            current_second_order_item += per_2nd_item
        per_2nd_item = list(set(current_second_order_item))
        per_mask_num_2nd_item = len(per_2nd_item)

        pad_per_2nd_item = per_2nd_item + [self.padding_number_items] * (
                setting.kshot_num * setting.kshot_second_num - len(per_2nd_item))

        current_third_order_user = []
        for each_kshot_item in per_2nd_item:
            candidate_third_order_user = self.neighbor_dict_item_list[each_kshot_item]

            if len(candidate_third_order_user) < setting.kshot_third_num:
                per_3rd_user = list(
                    np.random.choice(candidate_third_order_user, setting.kshot_third_num, replace=True))
            else:
                per_3rd_user = list(
                    np.random.choice(candidate_third_order_user, setting.kshot_third_num, replace=False))

            current_third_order_user += per_3rd_user
        per_3rd_user = list(set(current_third_order_user))
        per_mask_num_3rd_user = len(per_3rd_user)

        pad_per_3rd_user = per_3rd_user + [self.padding_number_users] * (
                setting.kshot_num * setting.kshot_third_num * setting.kshot_third_num - len(per_3rd_user))



        return per_1st_user, pad_per_2nd_item, pad_per_3rd_user, per_oracle_item_ebd, per_mask_num_2nd_item, per_mask_num_3rd_user

    ################## test set ###########################
    def get_test_users(self):
        test_user_list, test_user_pos_item, test_user_ground_truth = [], {}, {}
        with open(setting.downstream_support_file, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(',')
                user, item = int(arr[0]), int(arr[1])
                test_user_list.append(user)
                if user not in test_user_pos_item:
                    test_user_pos_item[user] = []
                    test_user_pos_item[user].append(item)
                else:
                    test_user_pos_item[user].append(item)

                if user not in test_user_ground_truth:
                    test_user_ground_truth[user] = []
                    test_user_ground_truth[user].append(item)
                else:
                    test_user_ground_truth[user].append(item)

                line = f.readline()

        with open(setting.downstream_query_file, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(',')
                user, item = int(arr[0]), int(arr[1])
                test_user_list.append(user)
                if user not in test_user_ground_truth:
                    test_user_ground_truth[user] = []
                    test_user_ground_truth[user].append(item)
                else:
                    test_user_ground_truth[user].append(item)

                line = f.readline()
        test_user_set = set(test_user_list)

        self.test_user_set = test_user_set
        self.test_user_pos_item = test_user_pos_item
        self.test_user_ground_truth = test_user_ground_truth

        # test user
        test_user_input, item_input = [], []
        u_1st_item, u_2nd_user, u_3rd_item, u_oracle_user_ebd, u_mask_num_2nd_user, u_mask_num_3rd_item = [], [], [], [], [], []
        i_1st_user, i_2nd_item, i_3rd_user, i_oracle_item_ebd, i_mask_num_2nd_item, i_mask_num_3rd_user = [], [], [], [], [], []

        for test_user in test_user_set:
            test_user_input.append(test_user)

            # user neighbor
            per_1st_i, per_2nd_u, per_3rd_i, per_oracle_u_ebd, per_mask_num_2nd_u, per_mask_num_3rd_i = self._get_user_neighbors(test_user)

            u_1st_item.append(per_1st_i)
            u_2nd_user.append(per_2nd_u)
            u_3rd_item.append(per_3rd_i)
            u_oracle_user_ebd.append(per_oracle_u_ebd)
            u_mask_num_2nd_user.append(per_mask_num_2nd_u)
            u_mask_num_3rd_item.append(per_mask_num_3rd_i)

        test_user_input = np.array(test_user_input)
        u_1st_item = np.array(u_1st_item)
        u_2nd_user = np.array(u_2nd_user)
        u_3rd_item = np.array(u_3rd_item)
        u_oracle_user_ebd = np.array(u_oracle_user_ebd)
        u_mask_num_2nd_user = np.array(u_mask_num_2nd_user)
        u_mask_num_3rd_item = np.array(u_mask_num_3rd_item)

        # all items
        for item_index in range(setting.num_items):
            item_input.append(item_index)

            # item neighbor
            per_1st_u, per_2nd_i, per_3rd_u, per_oracle_i_ebd, per_mask_num_2nd_i, per_mask_num_3rd_u = self._get_item_neighbors(item_index)

            i_1st_user.append(per_1st_u)
            i_2nd_item.append(per_2nd_i)
            i_3rd_user.append(per_3rd_u)
            i_oracle_item_ebd.append(per_oracle_i_ebd)
            i_mask_num_2nd_item.append(per_mask_num_2nd_i)
            i_mask_num_3rd_user.append(per_mask_num_3rd_u)

        item_input = np.array(item_input)
        i_1st_user = np.array(i_1st_user)
        i_2nd_item = np.array(i_2nd_item)
        i_3rd_user = np.array(i_3rd_user)
        i_oracle_item_ebd = np.array(i_oracle_item_ebd)
        i_mask_num_2nd_item = np.array(i_mask_num_2nd_item)
        i_mask_num_3rd_user = np.arary(i_mask_num_3rd_user)

        return test_user_input, test_user_pos_item, test_user_ground_truth, u_1st_item, u_2nd_user, u_3rd_item, u_oracle_user_ebd, u_mask_num_2nd_user, u_mask_num_3rd_item, i_1st_user, i_2nd_item, i_3rd_user, i_oracle_item_ebd, i_mask_num_2nd_item, i_mask_num_3rd_user

    def batch_gen_3rd_user_task(self, batches, i):
        return [(batches[r])[i] for r in range(7)]

    def batch_gen_3rd_item_task(self, batches, i):
        return [(batches[r])[i] for r in range(7)]


    def batch_gen_mix_user_task(self, batches, i):
        temp = min((i+1)*setting.batch_size, setting.num_users)

        return [(batches[r])[i * setting.batch_size: temp, :] for r in range(7)]

    def batch_gen_mix_item_task(self, batches, i):
        temp = min((i+1)*setting.batch_size, setting.num_items)
        return [(batches[r])[i * setting.batch_size: temp, :] for r in range(7)]


    def generate_all_users_neighbors(self):
        user_input, u_1st_item, u_2nd_user, u_3rd_item, u_oracle_user_ebd, u_mask_num_2nd_user, u_mask_num_3rd_item = [], [], [], [], [], [], []

        for user_index in range(self.num_users):
            per_1st_i, per_2nd_u, per_3rd_i, per_oracle_u_ebd, per_mask_num_2nd_u, per_mask_num_3rd_i = self._get_user_neighbors(user_index)

            user_input.append(user_index)
            u_1st_item.append(per_1st_i)
            u_2nd_user.append(per_2nd_u)
            u_3rd_item.append(per_3rd_i)
            u_oracle_user_ebd.append(per_oracle_u_ebd)
            u_mask_num_2nd_user.append(per_mask_num_2nd_u)
            u_mask_num_3rd_item.append(per_mask_num_3rd_i)

        user_input = np.array(user_input)

        u_1st_item = np.array(u_1st_item)
        u_2nd_user = np.array(u_2nd_user)
        u_3rd_item = np.array(u_3rd_item)
        u_oracle_user_ebd = np.array(u_oracle_user_ebd)
        u_mask_num_2nd_user = np.array(u_mask_num_2nd_user)
        u_mask_num_3rd_item = np.array(u_mask_num_3rd_item)

        user_input = np.reshape(user_input, (-1, 1))
        u_mask_num_2nd_user = np.reshape(u_mask_num_2nd_user, (-1, 1))
        u_mask_num_3rd_item = np.reshape(u_mask_num_3rd_item, (-1, 1))

        return user_input, u_1st_item, u_2nd_user, u_3rd_item, u_oracle_user_ebd, u_mask_num_2nd_user, u_mask_num_3rd_item

    def generate_all_items_neighbors(self):
        item_input, i_1st_user, i_2nd_item, i_3rd_user, i_oracle_item_ebd, i_mask_num_2nd_item, i_mask_num_3rd_user = [], [], [], [], [], [], []

        for item_index in range(self.num_items):
            per_1st_u, per_2nd_i, per_3rd_u, per_oracle_i_ebd, per_mask_num_2nd_i, per_mask_num_3rd_u = self._get_item_neighbors(item_index)

            item_input.append(item_index)
            i_1st_user.append(per_1st_u)
            i_2nd_item.append(per_2nd_i)
            i_3rd_user.append(per_3rd_u)
            i_oracle_item_ebd.append(per_oracle_i_ebd)
            i_mask_num_2nd_item.append(per_mask_num_2nd_i)
            i_mask_num_3rd_user.append(per_mask_num_3rd_u)

        item_input = np.array(item_input)
        i_1st_user = np.array(i_1st_user)
        i_2nd_item = np.array(i_2nd_item)
        i_3rd_user = np.array(i_3rd_user)
        i_oracle_item_ebd = np.array(i_oracle_item_ebd)
        i_mask_num_2nd_item = np.array(i_mask_num_2nd_item)
        i_mask_num_3rd_user = np.array(i_mask_num_3rd_user)

        item_input = np.reshape(item_input, (-1, 1))
        i_mask_num_2nd_item = np.reshape(i_mask_num_2nd_item, (-1, 1))
        i_mask_num_3rd_user = np.reshape(i_mask_num_3rd_user, (-1, 1))

        return item_input, i_1st_user, i_2nd_item, i_3rd_user, i_oracle_item_ebd, i_mask_num_2nd_item, i_mask_num_3rd_user



    ######### revised neighbors ############
    def get_train_instances_revised(self, train, epoch):

        final_user_1st_neighbor, final_user_2nd_neighbor, final_user_3rd_neighbor, final_item_1st_neighbor, final_item_2nd_neighbor, final_item_3rd_neighbor = \
            np.load('./r_user_task_k_shot_item.npy'), \
            np.load('./r_user_task_selected_input_2rd.npy'),\
            np.load('./r_user_task_selected_input_3rd.npy'), \
            np.load('./r_item_task_k_shot_user.npy'),\
            np.load('./r_item_task_selected_input_2rd.npy'),\
            np.load('./r_item_task_selected_input_3rd.npy')

        oracle_user_ebd = np.load(setting.oracle_user_ebd_path)
        oracle_item_ebd = np.load(setting.oracle_item_ebd_path)

        test_user_ground_truth = {}
        with open(setting.downstream_query_file, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(',')
                user, item = int(arr[0]), int(arr[1])
                if user not in test_user_ground_truth:
                    test_user_ground_truth[user] = []
                    test_user_ground_truth[user].append(item)
                else:
                    test_user_ground_truth[user].append(item)

                line = f.readline()
        with open(setting.downstream_meta_train_file, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(',')
                user, item = int(arr[0]), int(arr[1])
                if user not in test_user_ground_truth:
                    test_user_ground_truth[user] = []
                    test_user_ground_truth[user].append(item)
                else:
                    test_user_ground_truth[user].append(item)

                line = f.readline()

        np.random.seed(epoch)
        user_input, pos_item_input, neg_item_input = [], [], []
        u_1st_item, u_2nd_user, u_3rd_item, u_oracle_user_ebd, u_mask_num_2nd_user, u_mask_num_3rd_item = [], [], [], [], [], []
        ipos_1st_user, ipos_2nd_item, ipos_3rd_user, ipos_oracle_item_ebd, ipos_mask_num_2nd_item, ipos_mask_num_3rd_user = [], [], [], [], [], []
        ineg_1st_user, ineg_2nd_item, ineg_3rd_user, ineg_oracle_item_ebd, ineg_mask_num_2nd_item, ineg_mask_num_3rd_user = [], [], [], [], [], []

        num_users, num_items = setting.num_users, setting.num_items
        for (u, i) in train.keys():
            user_input.append(u)
            pos_item_input.append(i)
            j = np.random.randint(num_items)
            while (u, j) in train.keys() or j in test_user_ground_truth[u]:
                j = np.random.randint(num_items)
            neg_item_input.append(j)


            u_1st_item.append(final_user_1st_neighbor[u])
            u_2nd_user.append(final_user_2nd_neighbor[u])
            u_3rd_item.append(final_user_3rd_neighbor[u])
            u_oracle_user_ebd.append(oracle_user_ebd[u])

   
            ipos_1st_user.append(final_item_1st_neighbor[i])
            ipos_2nd_item.append(final_item_2nd_neighbor[i])
            ipos_3rd_user.append(final_item_3rd_neighbor[i])
            ipos_oracle_item_ebd.append(oracle_item_ebd[i])

            ineg_1st_user.append(final_item_1st_neighbor[j])
            ineg_2nd_item.append(final_item_2nd_neighbor[j])
            ineg_3rd_user.append(final_item_3rd_neighbor[j])
            ineg_oracle_item_ebd.append(oracle_item_ebd[j])

        user_input = np.array(user_input)
        u_1st_item = np.array(u_1st_item)
        u_2nd_user = np.array(u_2nd_user)
        u_3rd_item = np.array(u_3rd_item)
        u_oracle_user_ebd = np.array(u_oracle_user_ebd)

        pos_item_input = np.array(pos_item_input)
        ipos_1st_user = np.array(ipos_1st_user)
        ipos_2nd_item = np.array(ipos_2nd_item)
        ipos_3rd_user = np.array(ipos_3rd_user)
        ipos_oracle_item_ebd = np.array(ipos_oracle_item_ebd)

        neg_item_input = np.array(neg_item_input)
        ineg_1st_user = np.array(ineg_1st_user)
        ineg_2nd_item = np.array(ineg_2nd_item)
        ineg_3rd_user = np.array(ineg_3rd_user)
        ineg_oracle_item_ebd = np.array(ineg_oracle_item_ebd)


        return user_input, u_1st_item, u_2nd_user, u_3rd_item, u_oracle_user_ebd, u_mask_num_2nd_user, u_mask_num_3rd_item, pos_item_input, ipos_1st_user, ipos_2nd_item, ipos_3rd_user, ipos_oracle_item_ebd, ipos_mask_num_2nd_item, ipos_mask_num_3rd_user, neg_item_input, ineg_1st_user, ineg_2nd_item, ineg_3rd_user, ineg_oracle_item_ebd, ineg_mask_num_2nd_item, ineg_mask_num_3rd_user


    def get_test_users_revised(self):

        final_user_1st_neighbor, final_user_2nd_neighbor, final_user_3rd_neighbor, final_item_1st_neighbor, final_item_2nd_neighbor, final_item_3rd_neighbor = \
            np.load('./r_user_task_k_shot_item.npy'), \
            np.load('./r_user_task_selected_input_2rd.npy'),\
            np.load('./r_user_task_selected_input_3rd.npy'),\
            np.load('./r_item_task_k_shot_user.npy'),\
            np.load('./r_item_task_selected_input_2rd.npy'),\
            np.load('./r_item_task_selected_input_3rd.npy')

        oracle_user_ebd = np.load(setting.oracle_user_ebd_path)
        oracle_item_ebd = np.load(setting.oracle_item_ebd_path)

        test_user_list, test_user_pos_item, test_user_ground_truth = [], {}, {}
        with open(setting.downstream_support_file, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(',')
                user, item = int(arr[0]), int(arr[1])
                test_user_list.append(user)
                if user not in test_user_pos_item:
                    test_user_pos_item[user] = []
                    test_user_pos_item[user].append(item)
                else:
                    test_user_pos_item[user].append(item)

                if user not in test_user_ground_truth:
                    test_user_ground_truth[user] = []
                    test_user_ground_truth[user].append(item)
                else:
                    test_user_ground_truth[user].append(item)

                line = f.readline()

        with open(setting.downstream_query_file, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(',')
                user, item = int(arr[0]), int(arr[1])
                test_user_list.append(user)
                if user not in test_user_ground_truth:
                    test_user_ground_truth[user] = []
                    test_user_ground_truth[user].append(item)
                else:
                    test_user_ground_truth[user].append(item)

                line = f.readline()
        test_user_set = set(test_user_list)

        self.test_user_set = test_user_set
        self.test_user_pos_item = test_user_pos_item
        self.test_user_ground_truth = test_user_ground_truth

        # test user
        test_user_input, item_input = [], []
        u_1st_item, u_2nd_user, u_3rd_item, u_oracle_user_ebd, u_mask_num_2nd_user, u_mask_num_3rd_item = [], [], [], [], [], []
        i_1st_user, i_2nd_item, i_3rd_user, i_oracle_item_ebd, i_mask_num_2nd_item, i_mask_num_3rd_user= [], [], [], [], [], []

        for test_user in test_user_set:
            test_user_input.append(test_user)

          
            u_1st_item.append(final_user_1st_neighbor[test_user])
            u_2nd_user.append(final_user_2nd_neighbor[test_user])
            u_3rd_item.append(final_user_3rd_neighbor[test_user])
            u_oracle_user_ebd.append(oracle_user_ebd[test_user])

        test_user_input = np.array(test_user_input)
        u_1st_item = np.array(u_1st_item)
        u_2nd_user = np.array(u_2nd_user)
        u_3rd_item = np.array(u_3rd_item)
        u_oracle_user_ebd = np.array(u_oracle_user_ebd)

        # all items
        for item_index in range(setting.num_items):
            item_input.append(item_index)

            i_1st_user.append(final_item_1st_neighbor[item_index])
            i_2nd_item.append(final_item_2nd_neighbor[item_index])
            i_3rd_user.append(final_item_3rd_neighbor[item_index])
            i_oracle_item_ebd.append(oracle_item_ebd[item_index])

        item_input = np.array(item_input)
        i_1st_user = np.array(i_1st_user)
        i_2nd_item = np.array(i_2nd_item)
        i_3rd_user = np.array(i_3rd_user)
        i_oracle_item_ebd = np.array(i_oracle_item_ebd)

        return test_user_input, test_user_pos_item, test_user_ground_truth, u_1st_item, u_2nd_user, u_3rd_item, u_oracle_user_ebd, u_mask_num_2nd_user, u_mask_num_3rd_item, i_1st_user, i_2nd_item, i_3rd_user, i_oracle_item_ebd, i_mask_num_2nd_item, i_mask_num_3rd_user
