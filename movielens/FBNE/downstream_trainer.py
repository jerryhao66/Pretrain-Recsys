__author__ = 'haobowen'

import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from train_downstream_helper import *
from train_rl_helper import *  
import numpy as np
import tensorflow as tf
import setting
from FBNEConv import *
from time import time
import tqdm
import metrics as metrics
import heapq
import os
import gendata_fbne as data
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 4"




def evaluate(model, sess, test_batch):
    test_user_input, test_user_pos_item, test_user_ground_truth, test_u_1st_item, test_u_2nd_user, test_u_3rd_item, \
    test_u_intra_2nd, test_u_intra_3rd, test_u_oracle_user_ebd, test_u_mask_num_2nd_user, test_u_mask_num_3rd_item, \
    test_i_1st_user, test_i_2nd_item, test_i_3rd_user, test_i_intra_2nd, test_i_intra_3rd, test_i_oracle_item_ebd, test_i_mask_num_2nd_item, test_i_mask_num_3rd_user = test_batch

    precision, recall, ndcg, hit_ratio = [], [], [], []
    n_item_batchs = model.num_items // model.batch_size + 1

    test_user_length = range(len(test_user_input))
    i_count = 0
    test_item_list = np.reshape(np.array(range(model.num_items)), (-1, 1))
    for user_index in tqdm.tqdm(test_user_length):

        predcition = np.zeros(shape=(1, model.num_items))
        user = test_user_input[user_index]

        for i_batch_id in range(n_item_batchs):
            i_start = i_batch_id * model.batch_size
            i_end = min((i_batch_id + 1) * (model.batch_size), model.num_items)

            item_batch = range(i_start, i_end)


            feed_dict = {model.support_item_1st_: np.tile(test_u_1st_item[user_index, :], (i_end - i_start, 1)),
                         model.support_user_2nd_: np.tile(test_u_2nd_user[user_index, :], (i_end - i_start, 1)),
                         model.support_item_3rd: np.tile(test_u_3rd_item[user_index, :], (i_end - i_start, 1)),
                         model.inter_support_3rd_user: np.tile(test_u_intra_3rd[user_index, :], (i_end - i_start, 1)),
                         model.support_user_3rd_pos: test_i_3rd_user[i_start: i_end, :],
                         model.support_item_2nd_pos_: test_i_2nd_item[i_start: i_end, :],
                         model.support_user_1st_pos_: test_i_1st_user[i_start: i_end, :],
                         model.inter_support_3rd_item_pos: test_i_intra_3rd[i_start: i_end, :],
                         model.training_phrase_user_task: True,
                         model.training_phrase_item_task: True}

            predict_batch = sess.run([model.batch_predicts], feed_dict)  # [None, 1]
            predict_batch = np.reshape(np.array(predict_batch), (-1, len(item_batch)))

            predcition[:, i_start: i_end] = predict_batch
            i_count += predict_batch.shape[1]
        assert i_count == model.num_items
        i_count = 0

        # def test_one_user
        rating = list(np.reshape(predcition, (-1)))  # (num_items,)

        u = user
        training_items = test_user_pos_item[u]  # test support set
        user_pos_test = test_user_ground_truth[u]  # test ground_truth

        all_items = set(range(model.num_items))
        test_items = list(all_items - set(training_items))

        Ks = setting.Ks
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)

        precision.append(metrics.precision_at_k(r, Ks[0]))
        recall.append(metrics.recall_at_k(r, Ks[0], len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, Ks[0]))
        hit_ratio.append(metrics.hit_at_k(r, Ks[0]))
    return np.mean(np.array(precision)), np.mean(np.array(recall)), np.mean(np.array(ndcg)), np.mean(
        np.array(hit_ratio))


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    '''
    test_items = all_items - test_user_support_selected_items
    '''
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc


def revise_test_neighbors_item_task_downstream(name):
    print("revise all items high-order neighbors...")
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
            rl_saver = tf.train.Saver(max_to_keep=3)
            rl_saver.restore(sess3, tf.train.get_checkpoint_state(
                os.path.dirname(setting.checkpoint_path_rl_item_task + 'checkpoint')).model_checkpoint_path)


            downstream_data = DataLoader(setting.downstream_meta_train_file, setting.downstream_support_file)
            item_batches = downstream_data.generate_all_items_neighbors()

            num_batch = downstream_data.num_items // setting.batch_size
            all_target_ori_ebd, all_target_aggre_ebd, all_target_second_aggre_ebd, all_first_neigh_ori_ebd, all_second_neigh_ori_ebd, all_third_neigh_ori_ebd, all_intra_2nd_ori_ebd, all_intra_3rd_ori_ebd, all_first_order_reward1, all_first_and_second_order_reward2, all_former_third_order_reward3 = model.generate_batch_state_ebd_item_task_downstream(
                item_batches, downstream_data)

            r_target_item, r_k_shot_user, r_selected_input_2rd, r_selected_input_3rd, r_oracle_item_ebd = [], [], [], [], []
            for batch_index in tqdm.tqdm(range(num_batch)):

                b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_intra_2nd_item, b_intra_3rd_item, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user = downstream_data.batch_gen_mix_item_task(
                    item_batches, batch_index)

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

                b_target_ori_ebd = np.reshape(b_target_ori_ebd, (setting.batch_size, -1))

                max_num = b_second_neigh_ori_ebd.shape[1]
                max_num_3rd = b_third_neigh_ori_ebd.shape[1]

                original_reward_2rd = model.produce_original_reward_item_task_downstream(item_batches, num_batch,
                                                                                         downstream_data)
                original_reward_3rd = model.produce_original_reward_3rd_item_task_downstream(item_batches, num_batch,
                                                                                             downstream_data)

                # batch_original_reward = np.reshape(b_prob2, (-1))
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

                r_target_item.append(b_target_item)
                r_k_shot_user.append(b_k_shot_user)
                r_selected_input_2rd.append(selected_input_2rd)
                r_selected_input_3rd.append(selected_input_3rd)
                r_oracle_item_ebd.append(b_oracle_item_ebd)
            print(np.array(r_target_item).shape)
            print(np.array(r_k_shot_user).shape)
            print(np.array(r_selected_input_2rd).shape)
            print(np.array(r_selected_input_3rd).shape)
            
            # concatnate
            item_input, i_1st_user, i_2nd_item, i_3rd_user, i_intra_2nd, i_intra_3rd, i_oracle_item_ebd, i_mask_num_2nd_item, i_mask_num_3rd_user = item_batches
     
            trancated_item_input = item_input[num_batch * setting.batch_size:setting.num_items, :]
            trancated_item_input = np.reshape(trancated_item_input, (-1))
            r_target_item.append(trancated_item_input)
            for index in range(len(r_target_item)):
                if index == 0:
                    final_r_target_item = r_target_item[index]
                else:
                    temp = np.reshape(np.array(r_target_item[index]), (-1, 1))

                    final_r_target_item = np.concatenate([final_r_target_item, temp], 0)
            print(final_r_target_item.shape)

            trancated_i_1st_user = i_1st_user[num_batch * setting.batch_size:setting.num_items, :]
            r_k_shot_user.append(trancated_i_1st_user)
            for index in range(len(r_k_shot_user)):
                if index == 0:
                    final_i_1st_user = r_k_shot_user[index]
                else:
                    final_i_1st_user = np.concatenate([final_i_1st_user, r_k_shot_user[index]], 0)
            print(final_i_1st_user.shape)

            trancated_i_2nd_item = i_2nd_item[num_batch * setting.batch_size:setting.num_items, :]
            r_selected_input_2rd.append(trancated_i_2nd_item)
            for index in range(len(r_selected_input_2rd)):
                if index == 0:
                    final_i_selected_2rd = r_selected_input_2rd[index]
                else:
                    final_i_selected_2rd = np.concatenate([final_i_selected_2rd, r_selected_input_2rd[index]], 0)
            print(final_i_selected_2rd.shape)

            trancated_i_3rd_user = i_3rd_user[num_batch * setting.batch_size:setting.num_items, :]
            r_selected_input_3rd.append(trancated_i_3rd_user)
            for index in range(len(r_selected_input_3rd)):
                if index == 0:
                    final_i_selected_3rd = r_selected_input_3rd[index]
                else:
                    final_i_selected_3rd = np.concatenate([final_i_selected_3rd, r_selected_input_3rd[index]], 0)
            print(final_i_selected_3rd.shape)

        np.save('r_item_task_target_item.npy', np.array(final_r_target_item))
        np.save('r_item_task_k_shot_user.npy', np.array(final_i_1st_user))
        np.save('r_item_task_selected_input_2rd.npy', np.array(final_i_selected_2rd))
        np.save('r_item_task_selected_input_3rd.npy', np.array(final_i_selected_3rd))



def revise_test_neighbors_user_task_downstream(name):
    print("revise all users high-order neighbors...")
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

            downstream_data = DataLoader(setting.downstream_meta_train_file, setting.downstream_support_file)

            users_batches = downstream_data.generate_all_users_neighbors()

            num_batch = downstream_data.num_users // setting.batch_size
            all_target_ori_ebd, all_target_aggre_ebd, all_target_second_aggre_ebd, all_first_neigh_ori_ebd, all_second_neigh_ori_ebd, all_third_neigh_ori_ebd, all_intra_2nd_ori_ebd, all_intra_3rd_ori_ebd, all_first_order_reward1, all_first_and_second_order_reward2, all_former_third_order_reward3 = model.generate_batch_state_ebd_user_task_downstream(
                users_batches, downstream_data)

            r_target_user, r_k_shot_item, r_selected_input_2rd, r_selected_input_3rd, r_oracle_user_ebd = [], [], [], [], []
            for batch_index in tqdm.tqdm(range(num_batch)):
                b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_intra_2nd_user, b_intra_3rd_user, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item = downstream_data.batch_gen_mix_user_task(
                    users_batches, batch_index)

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

                original_reward_2rd = model.produce_original_reward_user_task_downstream(users_batches, num_batch,
                                                                                         downstream_data)
                original_reward_3rd = model.produce_original_reward_3rd_user_task_downstream(users_batches, num_batch,
                                                                                             downstream_data)

                batch_original_reward = np.reshape(b_prob3, (-1))
                b_target_ori_ebd = np.reshape(b_target_ori_ebd, (setting.batch_size, -1))


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

                r_target_user.append(b_target_user)
                r_k_shot_item.append(b_k_shot_item)
                r_selected_input_2rd.append(selected_input_2rd)
                r_selected_input_3rd.append(selected_input_3rd)
                r_oracle_user_ebd.append(b_oracle_user_ebd)

            # concatnate
            user_input, u_1st_item, u_2nd_user, u_3rd_item, u_intra_2nd, u_intra_3rd, u_oracle_user_ebd, u_mask_num_2nd_user, u_mask_num_3rd_item = users_batches
            trancated_user_input = user_input[num_batch * setting.batch_size:setting.num_users, :]
            trancated_user_input = np.reshape(trancated_user_input, (-1, 1))
            r_target_user.append(trancated_user_input)
            for index in range(len(r_target_user)):
                if index == 0:
                    final_r_target_user = r_target_user[index]
                else:
                    final_r_target_user = np.concatenate([final_r_target_user, r_target_user[index]], 0)
            print(final_r_target_user.shape)

            trancated_u_1st_item = u_1st_item[num_batch * setting.batch_size:setting.num_users, :]
            r_k_shot_item.append(trancated_u_1st_item)
            for index in range(len(r_k_shot_item)):
                if index == 0:
                    final_u_1st_item = r_k_shot_item[index]
                else:
                    final_u_1st_item = np.concatenate([final_u_1st_item, r_k_shot_item[index]], 0)
            print(final_u_1st_item.shape)

            trancated_u_2nd_user = u_2nd_user[num_batch * setting.batch_size:setting.num_users, :]
            r_selected_input_2rd.append(trancated_u_2nd_user)
            for index in range(len(r_selected_input_2rd)):
                if index == 0:
                    final_u_selected_2rd = r_selected_input_2rd[index]
                else:
                    final_u_selected_2rd = np.concatenate([final_u_selected_2rd, r_selected_input_2rd[index]], 0)
            print(final_u_selected_2rd.shape)

            trancated_u_3rd_item = u_3rd_item[num_batch * setting.batch_size:setting.num_users, :]
            r_selected_input_3rd.append(trancated_u_3rd_item)
            for index in range(len(r_selected_input_3rd)):
                if index == 0:
                    final_u_selected_3rd = r_selected_input_3rd[index]
                else:
                    final_u_selected_3rd = np.concatenate([final_u_selected_3rd, r_selected_input_3rd[index]], 0)
            print(final_u_selected_3rd.shape)

        np.save('r_user_task_target_user.npy', np.array(final_r_target_user))
        np.save('r_user_task_k_shot_item.npy', np.array(final_u_1st_item))
        np.save('r_user_task_selected_input_2rd.npy', np.array(final_u_selected_2rd))
        np.save('r_user_task_selected_input_3rd.npy', np.array(final_u_selected_3rd))



def train_downstream_task_revised(name):
   
    data = DataLoader(setting.downstream_meta_train_file, setting.downstream_support_file)
    train = data.train
    best_hr, best_loss = 0.0, 10.0
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

            print('start load train...')
            revise_test_neighbors_item_task_downstream(name)
            revise_test_neighbors_user_task_downstream(name)
            user_input, u_1st_item, u_2nd_user, u_3rd_item, u_intra_2nd, u_intra_3rd, u_oracle_user_ebd, u_mask_num_2nd_user, u_mask_num_3rd_item, \
            pos_item_input, ipos_1st_user, ipos_2nd_item, ipos_3rd_user, ipos_intra_2nd, ipos_intra_3rd, ipos_oracle_item_ebd, ipos_mask_num_2nd_item, ipos_mask_num_3rd_user, \
            neg_item_input, ineg_1st_user, ineg_2nd_item, ineg_3rd_user, ineg_intra_2nd, ineg_intra_3rd, ineg_oracle_item_ebd, ineg_mask_num_2nd_item, ineg_mask_num_3rd_user = data.get_train_instances_revised(
                train, 0)
            print('already load train...')

            batch_size = setting.batch_size
            batch_index = []
            for i in range(len(user_input)):
                if i % batch_size == 0:
                    batch_index.append(i)
            batch_index.append(len(user_input))

            print('start load test...')
            test_batches = data.get_test_users_revised()
            print('already load test...')
            for epoch in range(setting.epochs):
                train_loss = 0.0
                t1 = time()
                user_input, u_1st_item, u_2nd_user, u_3rd_item, u_intra_2nd, u_intra_3rd, u_oracle_user_ebd, u_mask_num_2nd_user, u_mask_num_3rd_item, \
                pos_item_input, ipos_1st_user, ipos_2nd_item, ipos_3rd_user, ipos_intra_2nd, ipos_intra_3rd, ipos_oracle_item_ebd, ipos_mask_num_2nd_item, ipos_mask_num_3rd_user, \
                neg_item_input, ineg_1st_user, ineg_2nd_item, ineg_3rd_user, ineg_intra_2nd, ineg_intra_3rd, ineg_oracle_item_ebd, ineg_mask_num_2nd_item, ineg_mask_num_3rd_user = data.get_train_instances_revised(
                    train, epoch)
                print(user_input.shape)
                print(pos_item_input.shape)
                print(neg_item_input.shape)
                for step in range(len(batch_index) - 1):
                    feed_dict = {model.support_item_1st_: u_1st_item[batch_index[step]:batch_index[step + 1], :],
                                 model.support_user_2nd_: u_2nd_user[batch_index[step]:batch_index[step + 1], :],
                                 model.support_item_3rd: u_3rd_item[batch_index[step]:batch_index[step + 1], :],
                                 model.inter_support_3rd_user: u_intra_3rd[batch_index[step]:batch_index[step + 1], :],
                                 model.support_user_3rd_pos: ipos_3rd_user[batch_index[step]: batch_index[step + 1], :],
                                 model.support_item_2nd_pos_: ipos_2nd_item[batch_index[step]: batch_index[step + 1],
                                                              :],
                                 model.support_user_1st_pos_: ipos_1st_user[batch_index[step]: batch_index[step + 1],
                                                              :],
                                 model.inter_support_3rd_item_pos: ipos_intra_3rd[
                                                                   batch_index[step]: batch_index[step + 1], :],
                                 model.support_user_3rd_neg: ineg_3rd_user[batch_index[step]: batch_index[step + 1], :],
                                 model.support_item_2nd_neg_: ineg_2nd_item[batch_index[step]: batch_index[step + 1],
                                                              :],
                                 model.support_user_1st_neg_: ineg_1st_user[batch_index[step]: batch_index[step + 1],
                                                              :],
                                 model.inter_support_3rd_item_neg: ineg_intra_3rd[
                                                                   batch_index[step]: batch_index[step + 1], :],
                                 model.training_phrase_user_task: True,
                                 model.training_phrase_item_task: True}

                    t, loss = sess3.run([model.third_opt, model.third_loss], feed_dict=feed_dict)
                    train_loss += loss
                train_loss = train_loss / len(batch_index)
                print(train_loss)

                if epoch % 1 == 0:
                    precision, recall, ndcg, hr = evaluate(model, sess3, test_batches)
                    t2 = time()
                    print(
                        'Iteration %d [%.1f s]: precision = %.4f, recall = %.4f, ndcg = %.4f, hr = %.4f, loss = %.4f [%.1f s]'
                        % (epoch, t2 - t1, precision, recall, ndcg, hr, loss, time() - t2))


if __name__ == '__main__':
    name = 'FBNE'
    train_downstream_task_revised(name)
