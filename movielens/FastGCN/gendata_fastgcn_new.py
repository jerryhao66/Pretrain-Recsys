
import setting
import numpy as np


num_users = setting.num_users 
num_items = setting.num_items 

def id_to_useritemid(num_users, num_items):
    all_dict = {}
    for id in range(num_users):
        all_dict[id] = id
    for id in range(num_items):
        all_dict[id + num_users] = id
    return all_dict

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

        self.oracle_user_ebd = np.load(setting.oracle_user_ebd_path)
        self.oracle_item_ebd = np.load(setting.oracle_item_ebd_path)

    #### user task ####

    def get_positive_instances_user_task(self, data, all_dict):
        target_user, k_shot_item, second_order_uesrs, oracle_user_ebd, mask_num_second_order_user, third_order_items, mask_num_third_order_item = [], [], [], [], [], [], []
        target_user_first_order, target_user_second_order, target_user_third_order = {}, {}, {}
        with open('./fastgcn_first_order_user.txt', 'r') as f:
            count = -1
            line = f.readline()
            while line!="" and line!=None:
                count +=1
                arr = line.strip().split(' ')
                target_user_first_order[count] = []
                for neighbor in arr:
                    target_user_first_order[count].append(int(neighbor))
                line = f.readline()

        with open('./fastgcn_second_order_user.txt', 'r') as f:
            count = -1
            line = f.readline()
            while line!="" and line!=None:
                count +=1
                arr = line.strip().split(' ')
                target_user_second_order[count] = []
                for neighbor in arr:
                    target_user_second_order[count].append(int(neighbor))
                line = f.readline()


        with open('./fastgcn_third_order_user.txt', 'r') as f:
            count = -1
            line = f.readline()
            while line!="" and line!=None:
                count +=1
                arr = line.strip().split(' ')
                target_user_third_order[count] = []
                for neighbor in arr:
                    target_user_third_order[count].append(int(neighbor))
                line = f.readline()


        for user in range(setting.num_users):
            target_user.append(user)
            k_shot_item.append(target_user_first_order[user])
            second_order_uesrs.append(target_user_second_order[user])
            third_order_items.append(target_user_third_order[user])
            oracle_user_ebd.append(data.oracle_user_ebd[user])
            mask_num_second_order_user.append(len(target_user_second_order[user]))
            mask_num_third_order_item.append(len(target_user_third_order[user]))



        return target_user, k_shot_item, second_order_uesrs, third_order_items, oracle_user_ebd, mask_num_second_order_user, mask_num_third_order_item


    ##### item task #####
    def get_positive_instances_item_task(self, data, all_dict):
        target_item, k_shot_user, second_order_items, oracle_item_ebd, mask_num_second_order_item, third_order_users, mask_num_third_order_user = [], [], [], [], [], [], []
        target_item_first_order, target_item_second_order, target_item_third_order = {}, {}, {}
        with open('./fastgcn_first_order_item.txt', 'r') as f:
            count = -1
            line = f.readline()
            while line!="" and line!=None:
                count +=1
                arr = line.strip().split(' ')
                target_item_first_order[count] = []
                for neighbor in arr:
                    target_item_first_order[count].append(int(neighbor))
                line = f.readline()


        with open('./fastgcn_second_order_item.txt', 'r') as f:
            count = -1
            line = f.readline()
            while line!="" and line!=None:
                count +=1
                arr = line.strip().split(' ')
                target_item_second_order[count] = []
                for neighbor in arr:
                    target_item_second_order[count].append(int(neighbor))
                line = f.readline()


        with open('./fastgcn_third_order_item.txt', 'r') as f:
            count = -1
            line = f.readline()
            while line!="" and line!=None:
                count +=1
                arr = line.strip().split(' ')
                target_item_third_order[count] = []
                for neighbor in arr:
                    target_item_third_order[count].append(int(neighbor))
                line = f.readline()


        for item in range(setting.num_items):
            target_item.append(item)
            k_shot_user.append(target_item_first_order[item])
            second_order_items.append(target_item_second_order[item])
            third_order_users.append(target_item_third_order[item])
            oracle_item_ebd.append(data.oracle_user_ebd[item])
            mask_num_second_order_item.append(len(target_item_second_order[item]))
            mask_num_third_order_user.append(len(target_item_third_order[item]))



        return target_item, k_shot_user, second_order_items, third_order_users, oracle_item_ebd, mask_num_second_order_item, mask_num_third_order_user

def split_batch_item(target_item, k_shot_user, second_order_items, third_order_users, oracle_item_ebd, mask_num_second_order_item, mask_num_third_order_user, i):

    begin_index = i * setting.fastgcn_batch_size
    end_index = min(begin_index + setting.fastgcn_batch_size, setting.num_items)

    batch_target_item = target_item[begin_index: end_index]
    batch_kshot_user = k_shot_user[begin_index: end_index]
    batch_2nd_item = second_order_items[begin_index: end_index]
    batch_3rd_user = third_order_users[begin_index: end_index]
    batch_oracle_item_ebd = oracle_item_ebd[begin_index: end_index]
    batch_mask_num_2nd_item = mask_num_second_order_item[begin_index: end_index]
    batch_mask_num_3rd_user = mask_num_third_order_user[begin_index: end_index]

    return batch_target_item, batch_kshot_user, batch_2nd_item, batch_3rd_user, batch_oracle_item_ebd, batch_mask_num_2nd_item, batch_mask_num_3rd_user

def split_batch_user(target_user, k_shot_item, second_order_uesrs, third_order_items, oracle_user_ebd, mask_num_second_order_user, mask_num_third_order_item, i):

    begin_index = i * setting.fastgcn_batch_size
    end_index = min(begin_index + setting.fastgcn_batch_size, setting.num_users)

    batch_target_user = target_user[begin_index: end_index]
    batch_kshot_item = k_shot_item[begin_index: end_index]
    batch_2nd_user = second_order_uesrs[begin_index: end_index]
    batch_3rd_item = third_order_items[begin_index: end_index]
    batch_oracle_user_ebd = oracle_user_ebd[begin_index: end_index]
    batch_mask_num_2nd_user = mask_num_second_order_user[begin_index: end_index]
    batch_mask_num_3rd_item = mask_num_third_order_item[begin_index: end_index]

    return batch_target_user, batch_kshot_item, batch_2nd_user, batch_3rd_item, batch_oracle_user_ebd, batch_mask_num_2nd_user, batch_mask_num_3rd_item
