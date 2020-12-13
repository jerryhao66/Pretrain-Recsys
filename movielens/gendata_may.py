import numpy as np
import setting


class Dataset(object):
    '''
    Load the original rating file
    '''

    def __init__(self, data_path):
        self.num_items = setting.num_items
        self.num_users = setting.num_users
        self.batch_size = setting.batch_size
        self.kshot_num = setting.kshot_num
        self.kshot_second_num = setting.kshot_second_num
        self.kshot_third_num = setting.kshot_third_num
        self.padding_number_items = self.num_items
        self.padding_number_users = self.num_users

        self.oracle_uesr_ebd = np.load(setting.oracle_user_ebd_path)
        self.oracle_item_ebd = np.load(setting.oracle_item_ebd_path)

        self.neighbor_dict_user_list, self.neighbor_dict_item_list = self.load_original_rating_file_as_list(
            data_path)
        self.generate_oracle_users_and_items(data_path)

    def load_original_rating_file_as_list(self, filename):
        neighbor_dict_user_list, neighbor_dict_item_list = {}, {}
        with open(filename, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(',')
                user, item = int(arr[0]), int(arr[1])
                if user not in neighbor_dict_user_list:
                    neighbor_dict_user_list[user] = []
                    neighbor_dict_user_list[user].append(item)
                else:
                    neighbor_dict_user_list[user].append(item)

                if item not in neighbor_dict_item_list:
                    neighbor_dict_item_list[item] = []
                    neighbor_dict_item_list[item].append(user)
                else:
                    neighbor_dict_item_list[item].append(user)
                line = f.readline()

        # padding, if the number of user and item is not in range(num_items) and range(num_users)
        for user in range(self.num_users):
            if user not in neighbor_dict_user_list.keys():
                neighbor_dict_user_list[user] = []
                neighbor_dict_user_list[user].append(self.num_items)  # padding

        for item in range(self.num_items):
            if item not in neighbor_dict_item_list.keys():
                neighbor_dict_item_list[item] = []
                neighbor_dict_item_list[item].append(self.num_users)

        return neighbor_dict_user_list, neighbor_dict_item_list

    ##########   generate few-shot positive instances      ##########
    '''      for each user, randomly select k-shot items
                 and maxnumber second order 3*k-shot users
    '''

    ##########                                             ##########
    def generate_oracle_users_and_items(self, filename):
        oracle_user_list, oracle_item_list = [], []
        with open(filename, 'r') as f:
            line = f.readline()
            while line != "" and line != None:
                arr = line.strip().split(',')
                user, item = int(arr[0]), int(arr[1])
                oracle_user_list.append(user)
                oracle_item_list.append(item)

                line = f.readline()

            oracle_user_set = set(oracle_user_list)
            oracle_item_set = set(oracle_item_list)
            self.oracle_user_list = list(oracle_user_set)
            self.oracle_item_list = list(oracle_item_set)
            self.oracle_num_users = len(oracle_user_set)
            self.oracle_num_items = len(oracle_item_set)


    '''
    mix-user-task
    '''

    def get_positive_instances_user_task(self, random_seed):
        '''
        uesr-task
        '''
        np.random.seed(random_seed)
        batch_num = self.oracle_num_users // self.batch_size + 1
        target_user, k_shot_item, second_order_uesrs, oracle_user_ebd, mask_num_second_order_user, third_order_items, mask_num_third_order_item = [], [], [], [], [], [], []
        for batch in range(batch_num):
            b_target_u, b_k_shot_item, b_2nd_order_u, b_3rd_order_i, b_oracle_u_ebd, b_mask_num_2nd_u, b_mask_num_3rd_i = self._get_positive_batch_user_task(
                batch)
            target_user.append(b_target_u)
            k_shot_item.append(b_k_shot_item)
            second_order_uesrs.append(b_2nd_order_u)
            oracle_user_ebd.append(b_oracle_u_ebd)
            mask_num_second_order_user.append(b_mask_num_2nd_u)

            third_order_items.append(b_3rd_order_i)
            mask_num_third_order_item.append(b_mask_num_3rd_i)

        return target_user, k_shot_item, second_order_uesrs, third_order_items, oracle_user_ebd, mask_num_second_order_user, mask_num_third_order_item

    def _get_positive_batch_user_task(self, i):
        batch_target_user, batch_kshot_item, batch_2nd_user, batch_oracle_user_ebd, mask_num_2nd_user, batch_3rd_item, mask_num_3rd_item = [], [], [], [], [], [], []

        begin_index = i * self.batch_size
        end_index = min(begin_index + self.batch_size, self.oracle_num_users)
        for per_user_index in range(begin_index, end_index):
            target_user_id = self.oracle_user_list[per_user_index]

            per_oracle_user_ebd = self.oracle_uesr_ebd[target_user_id]
            sample_kshot_item = np.random.choice(self.neighbor_dict_user_list[target_user_id], self.kshot_num,
                                                 replace=False)
            current_second_order_user = []
            for each_kshot_item in sample_kshot_item:
                candidate_second_order_user = self.neighbor_dict_item_list[each_kshot_item]


                if target_user_id in candidate_second_order_user:
                    candidate_second_order_user.remove(target_user_id)

                if len(candidate_second_order_user) == 0:
                    candidate_second_order_user.append(target_user_id)

                if len(candidate_second_order_user) < self.kshot_second_num:
                    temp_second_order_user = list(
                        np.random.choice(candidate_second_order_user, self.kshot_second_num, replace=True))
                else:
                    temp_second_order_user = list(
                        np.random.choice(candidate_second_order_user, self.kshot_second_num, replace=False))

                current_second_order_user += temp_second_order_user
            temp_second_order_user = list(set(current_second_order_user))

            current_third_order_item = []
            for each_kshot_3rd_user in temp_second_order_user:
                candidate_third_order_item = self.neighbor_dict_user_list[each_kshot_3rd_user]
                if len(candidate_third_order_item) < self.kshot_third_num:
                    temp_third_order_item = list(
                        np.random.choice(candidate_third_order_item, self.kshot_third_num, replace=True))
                else:
                    temp_third_order_item = list(
                        np.random.choice(candidate_third_order_item, self.kshot_third_num, replace=False))
                current_third_order_item += temp_third_order_item
            temp_third_order_item = list(set(current_third_order_item))

            batch_target_user.append(target_user_id)
            batch_kshot_item.append(sample_kshot_item)
            batch_2nd_user.append(temp_second_order_user)
            batch_oracle_user_ebd.append(per_oracle_user_ebd)
            mask_num_2nd_user.append(len(temp_second_order_user))

            batch_3rd_item.append(temp_third_order_item)
            mask_num_3rd_item.append(len(temp_third_order_item))

        batch_2nd_user_input = self._add_mask(self.padding_number_users, batch_2nd_user, max(mask_num_2nd_user))
        batch_3rd_item_input = self._add_mask(self.padding_number_items, batch_3rd_item, max(mask_num_3rd_item))

        return batch_target_user, batch_kshot_item, batch_2nd_user_input, batch_3rd_item_input, batch_oracle_user_ebd, mask_num_2nd_user, mask_num_3rd_item

    '''
    mix item-task
    '''

    def get_positive_instances_item_task(self, random_seed):
        np.random.seed(random_seed)
        batch_num = self.oracle_num_items // self.batch_size + 1
        target_item, k_shot_user, second_order_items, oracle_item_ebd, mask_num_second_order_item, third_order_users, mask_num_third_order_user = [], [], [], [], [], [], []
        for batch in range(batch_num):
            b_target_i, b_k_shot_user, b_2nd_order_i, b_3rd_order_u, b_oracle_i_ebd, b_mask_num_2nd_i, b_mask_num_3rd_u = self._get_positive_batch_item_task(
                batch)
            target_item.append(b_target_i)
            k_shot_user.append(b_k_shot_user)
            second_order_items.append(b_2nd_order_i)
            oracle_item_ebd.append(b_oracle_i_ebd)
            mask_num_second_order_item.append(b_mask_num_2nd_i)

            third_order_users.append(b_3rd_order_u)
            mask_num_third_order_user.append(b_mask_num_3rd_u)

        return target_item, k_shot_user, second_order_items, third_order_users, oracle_item_ebd, mask_num_second_order_item, mask_num_third_order_user

    def _get_positive_batch_item_task(self, i):
        batch_target_item, batch_kshot_user, batch_2nd_item, batch_oracle_item_ebd, mask_num_2nd_item, batch_3rd_user, mask_num_3rd_user = [], [], [], [], [], [], []

        begin_index = i * self.batch_size
        end_index = min(begin_index + self.batch_size, self.oracle_num_items)
        for per_item_index in range(begin_index, end_index):
            target_item_id = self.oracle_item_list[per_item_index]

            per_oracle_item_ebd = self.oracle_item_ebd[target_item_id]
            if len(self.neighbor_dict_item_list[target_item_id]) < self.kshot_num:
                sample_kshot_user = np.random.choice(self.neighbor_dict_item_list[target_item_id], self.kshot_num,
                                                     replace=True)
            else:
                sample_kshot_user = np.random.choice(self.neighbor_dict_item_list[target_item_id], self.kshot_num,
                                                     replace=False)

            current_second_order_item = []
            for each_kshot_user in sample_kshot_user:
                candidate_second_order_item = self.neighbor_dict_user_list[each_kshot_user]
                if target_item_id in candidate_second_order_item:
                    candidate_second_order_item.remove(target_item_id)
                if len(candidate_second_order_item) == 0:
                    candidate_second_order_item.append(target_item_id)

                if len(candidate_second_order_item) < self.kshot_second_num:
                    temp_second_order_item = list(
                        np.random.choice(candidate_second_order_item, self.kshot_second_num, replace=True))
                else:
                    temp_second_order_item = list(
                        np.random.choice(candidate_second_order_item, self.kshot_second_num, replace=False))
                current_second_order_item += temp_second_order_item
            temp_second_order_item = list(set(current_second_order_item))

            current_third_order_user = []
            for each_kshot_3rd_item in temp_second_order_item:
                candidate_third_order_user = self.neighbor_dict_item_list[each_kshot_3rd_item]
                if len(candidate_third_order_user) < self.kshot_third_num:
                    temp_third_order_user = list(
                        np.random.choice(candidate_third_order_user, self.kshot_third_num, replace=True))
                else:
                    temp_third_order_user = list(
                        np.random.choice(candidate_third_order_user, self.kshot_third_num, replace=False))
                current_third_order_user += temp_third_order_user
            temp_third_order_user = list(set(current_third_order_user))

            batch_target_item.append(target_item_id)
            batch_kshot_user.append(sample_kshot_user)
            batch_2nd_item.append(temp_second_order_item)
            batch_oracle_item_ebd.append(per_oracle_item_ebd)
            mask_num_2nd_item.append(len(temp_second_order_item))

            batch_3rd_user.append(temp_third_order_user)
            mask_num_3rd_user.append(len(temp_third_order_user))

        batch_2nd_item_input = self._add_mask(self.padding_number_items, batch_2nd_item, max(mask_num_2nd_item))
        batch_3rd_user_input = self._add_mask(self.padding_number_users, batch_3rd_user, max(mask_num_3rd_user))

        return batch_target_item, batch_kshot_user, batch_2nd_item_input, batch_3rd_user_input, batch_oracle_item_ebd, mask_num_2nd_item, mask_num_3rd_user

    def _add_mask(self, feature_mask, features, num_max):

        # uniformalize the length of each batch
        for i in range(len(features)):
            features[i] = features[i] + [feature_mask] * (num_max - len(features[i]))
        return features

    def batch_gen_mix_user_task(self, batches, i):

        return [(batches[r])[i] for r in range(5)]

    def batch_gen_3rd_user_task(self, batches, i):
        return [(batches[r])[i] for r in range(7)]

    def batch_gen_mix_item_task(self, batches, i):

        return [(batches[r])[i] for r in range(5)]

    def batch_gen_3rd_item_task(self, batches, i):
        return [(batches[r])[i] for r in range(7)]


###########################      restore           #######################
'''
input: target is item, input k-shot users
'''


def generate_all_item_dict(all_rating):
    np.random.seed(0)
    all_item_dict = {}
    with open(all_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if item not in all_item_dict:
                all_item_dict[item] = []
                all_item_dict[item].append(user)
            else:
                all_item_dict[item].append(user)
            line = f.readline()

    all_support_item_dict = {}

    for each_item in all_item_dict.keys():
        all_support_item_dict[each_item] = []
        if len(all_item_dict[each_item]) >= setting.support_num:
            select_instance = np.random.choice(all_item_dict[each_item], setting.support_num, replace=False)
        else:
            select_instance = np.random.choice(all_item_dict[each_item], setting.support_num, replace=True)
        for user in select_instance:
            all_support_item_dict[each_item].append(user)

    return all_support_item_dict


def generate_meta_all_item_set(all_support_item_dict):
    all_item_id, all_support_set_user = [], []
    for item_key in all_support_item_dict.keys():
        pos_user = all_support_item_dict[item_key]

        all_item_id.append(item_key)
        all_support_set_user.append(pos_user)

    all_item_num_instance = len(all_item_id)

    all_item_id = np.array(all_item_id)
    all_support_set_user = np.array(all_support_set_user)

    return all_item_id, all_support_set_user, all_item_num_instance


'''
input: target is user, input k-shot items
'''


def generate_all_user_dict(all_rating):
    np.random.seed(0)
    all_user_dict = {}
    with open(all_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if user not in all_user_dict:
                all_user_dict[user] = []
                all_user_dict[user].append(item)
            else:
                all_user_dict[user].append(item)
            line = f.readline()

    all_support_user_dict = {}
    for each_user in all_user_dict.keys():
        all_support_user_dict[each_user] = []
        select_instance = np.random.choice(all_user_dict[each_user], setting.support_num, replace=False)
        for item in select_instance:
            all_support_user_dict[each_user].append(item)
    return all_support_user_dict


def generate_meta_all_user_set(all_support_user_dict):
    all_user_id, all_support_set_item = [], []
    for user_key in all_support_user_dict.keys():
        pos_item = all_support_user_dict[user_key]
        all_user_id.append(user_key)
        all_support_set_item.append(pos_item)

    all_num_instance = len(all_user_id)
    all_user_id = np.array(all_user_id)
    all_support_set_item = np.array(all_support_set_item)

    return all_user_id, all_support_set_item, all_num_instance


def batch_gen_task(batches, i, batch_size, num_instances):
    '''
    apply to train or valid dataset
    '''
    start_index = i * batch_size
    end_index = min(start_index + batch_size, num_instances)

    return [(batches[r])[start_index:end_index] for r in range(2)]



'''
load ground truth
'''


def load_target_user_embedding(user_ebd_path):
    user_embedding = np.load(user_ebd_path)

    return user_embedding


def load_target_item_embedding(item_ebd_path):
    item_embedding = np.load(item_ebd_path)

    return item_embedding


'''
user task: personalized recommendation
'''


def generate_user_dict_valid(valid_rating):
    valid_user_dict = {}
    with open(valid_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if user not in valid_user_dict:
                valid_user_dict[user] = []
                valid_user_dict[user].append(item)
            else:
                valid_user_dict[user].append(item)

            line = f.readline()

    np.random.seed(0)
    valid_support_user_dict = {}
    for valid_user in valid_user_dict.keys():
        valid_support_user_dict[valid_user] = []
        select_instance = np.random.choice(valid_user_dict[valid_user], setting.support_num, replace=False)
        for item in select_instance:
            valid_support_user_dict[valid_user].append(item)

    return valid_support_user_dict


def generate_user_dict_train(train_rating):
    train_user_dict = {}
    with open(train_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if user not in train_user_dict:
                train_user_dict[user] = []
                train_user_dict[user].append(item)
            else:
                train_user_dict[user].append(item)

            line = f.readline()

    train_support_user_dict = {}

    for train_user in train_user_dict.keys():
        train_support_user_dict[train_user] = []
        select_instance = np.random.choice(train_user_dict[train_user], setting.support_num, replace=False)
        for item in select_instance:
            train_support_user_dict[train_user].append(item)

    return train_support_user_dict


def generate_meta_train_user_set(train_support_user_dict):
    '''
    generate meta training/valid set
    '''
    target_user_ebd = load_target_user_embedding(setting.oracle_user_ebd_path)

    train_user_id, train_support_set_item, train_target_user = [], [], []
    for user_key in train_support_user_dict.keys():
        pos_item = train_support_user_dict[user_key]
        user_ebd = target_user_ebd[user_key]

        train_user_id.append(user_key)
        train_support_set_item.append(pos_item)
        train_target_user.append(user_ebd)

    train_num_instance = len(train_user_id)

    train_user_id = np.array(train_user_id)
    train_support_set_item = np.array(train_support_set_item)
    train_target_user = np.array(train_target_user)

    return train_user_id, train_support_set_item, train_target_user, train_num_instance


def generate_meta_valid_user_set(valid_support_user_dict):
    target_user_ebd = load_target_user_embedding(setting.oracle_user_ebd_path)

    valid_user_id, valid_support_set_item, valid_target_user = [], [], []
    for user_key in valid_support_user_dict.keys():
        pos_item = valid_support_user_dict[user_key]
        user_ebd = target_user_ebd[user_key]

        valid_user_id.append(user_key)
        valid_support_set_item.append(pos_item)
        valid_target_user.append(user_ebd)

    valid_num_instance = len(valid_user_id)

    return valid_user_id, valid_support_set_item, valid_target_user, valid_num_instance


def batch_gen_user_task(batches, i, batch_size):
    '''
    apply to train or valid dataset
    '''
    start_index = i * batch_size
    end_index = start_index + batch_size

    return [(batches[r])[start_index:end_index] for r in range(3)]


#################################################################
##########       item-task                         ##############

def generate_item_dict_valid(valid_rating):
    valid_item_dict = {}
    with open(valid_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if item not in valid_item_dict:
                valid_item_dict[item] = []
                valid_item_dict[item].append(user)
            else:
                valid_item_dict[item].append(user)

            line = f.readline()

    valid_support_item_dict = {}
    np.random.seed(0)
    for valid_item in valid_item_dict.keys():
        valid_support_item_dict[valid_item] = []
        if len(valid_item_dict[valid_item]) >= setting.support_num:
            select_instance = np.random.choice(valid_item_dict[valid_item], setting.support_num, replace=False)
        else:
            select_instance = np.random.choice(valid_item_dict[valid_item], setting.support_num, replace=True)
        for user in select_instance:
            valid_support_item_dict[valid_item].append(user)

    return valid_support_item_dict


def generate_item_dict_train(train_rating):
    train_item_dict = {}
    with open(train_rating, 'r') as f:
        line = f.readline()
        while line != "" and line != None:
            arr = line.strip().split(',')
            user, item = int(arr[0]), int(arr[1])
            if item not in train_item_dict:
                train_item_dict[item] = []
                train_item_dict[item].append(user)
            else:
                train_item_dict[item].append(user)

            line = f.readline()

    train_support_item_dict = {}

    for train_item in train_item_dict.keys():
        train_support_item_dict[train_item] = []
        if len(train_item_dict[train_item]) < setting.support_num:
            select_instance = np.random.choice(train_item_dict[train_item], setting.support_num, replace=True)
        else:
            select_instance = np.random.choice(train_item_dict[train_item], setting.support_num, replace=False)
        for user in select_instance:
            train_support_item_dict[train_item].append(user)

    return train_support_item_dict


def generate_meta_train_item_set(train_support_item_dict):
    '''
    generate meta training/valid set
    '''
    target_item_ebd = load_target_user_embedding(setting.oracle_item_ebd_path)

    train_item_id, train_support_set_user, train_target_item = [], [], []
    for item_key in train_support_item_dict.keys():
        pos_user = train_support_item_dict[item_key]
        item_ebd = target_item_ebd[item_key]

        train_item_id.append(item_key)
        train_support_set_user.append(pos_user)
        train_target_item.append(item_ebd)

    train_num_instance = len(train_item_id)

    train_item_id = np.array(train_item_id)
    train_support_set_user = np.array(train_support_set_user)
    train_target_item = np.array(train_target_item)

    return train_item_id, train_support_set_user, train_target_item, train_num_instance


def generate_meta_valid_item_set(valid_support_item_dict):
    target_item_ebd = load_target_user_embedding(setting.oracle_item_ebd_path)

    valid_item_id, valid_support_set_user, valid_target_item = [], [], []
    for item_key in valid_support_item_dict.keys():
        pos_user = valid_support_item_dict[item_key]
        item_ebd = target_item_ebd[item_key]

        valid_item_id.append(item_key)
        valid_support_set_user.append(pos_user)
        valid_target_item.append(item_ebd)

    valid_num_instance = len(valid_item_id)

    return valid_item_id, valid_support_set_user, valid_target_item, valid_num_instance


def batch_gen_item_task(batches, i, batch_size):
    '''
    apply to train or valid dataset
    '''
    start_index = i * batch_size
    end_index = start_index + batch_size

    return [(batches[r])[start_index:end_index] for r in range(3)]

