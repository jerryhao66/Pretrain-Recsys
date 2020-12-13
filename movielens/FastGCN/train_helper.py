__author__ = 'haobowen'

import tensorflow as tf
import logging
import os
import numpy as np
import setting
from time import time
import tqdm
from scipy.stats import pearsonr
from unit import *
import gendata_fastgcn_new as gfn



######################################  fastgcn new #################################################
'''
uesr_task
'''


def training_user_task_fastgcnnew(model, sess):
    best_loss = 0
    saver = tf.train.Saver()
    ts_u = gfn.Dataset(setting.oracle_training_file_user_task)

    vs_u = gfn.Dataset(setting.oracle_valid_file_user_task)

    num_users = setting.num_users  
    num_items = setting.num_items  
    all_dict = gfn.id_to_useritemid(num_users, num_items)

    train_batches = ts_u.get_positive_instances_user_task(ts_u, all_dict)
    valid_batches = vs_u.get_positive_instances_user_task(vs_u, all_dict)



    num_batch = int(len(train_batches[0])) // setting.fastgcn_batch_size
    batch_index = range(num_batch)
    valid_num_batch = int(len(valid_batches[0])) // setting.fastgcn_batch_size
    valid_batch_index = range(valid_num_batch)

    for epoch_count in range(setting.user_epoch):
        train_begin = time()
        training_batch_user_task_fastgcnnew(batch_index, model, sess, train_batches, True)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_user_task_fastgcnnew(batch_index, model, sess, train_batches, True)
            loss_time = time() - loss_begin
            eval_begin = time()
            cosine = evaluate_user_task_fastgcnnew(valid_batch_index, model, sess, valid_batches, True)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test cosine value is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, cosine))

            if cosine < best_loss:
                best_loss = cosine
                saver.save(sess, setting.checkpoint_path_user_task, global_step=epoch_count)

        ts_u = gfn.Dataset(setting.oracle_training_file_user_task)

        train_batches = ts_u.get_positive_instances_user_task(ts_u, all_dict)


def evaluate_user_task_fastgcnnew(valid_batch_index, model, sess, valid_data, is_training):
    valid_target_user, valid_k_shot_item, valid_second_order_uesrs, valid_third_order_items, valid_oracle_user_ebd, valid_mask_num_second_order_user, valid_mask_num_third_order_item = valid_data
    evaluate_loss = 0.0
    for index in tqdm.tqdm(valid_batch_index):
        batch_target_user, batch_kshot_item, batch_2nd_user, batch_3rd_item, batch_oracle_user_ebd, batch_mask_num_2nd_user, batch_mask_num_3rd_item = gfn.split_batch_user(
            valid_target_user, valid_k_shot_item, valid_second_order_uesrs, valid_third_order_items,
            valid_oracle_user_ebd, valid_mask_num_second_order_user, valid_mask_num_third_order_item, index)

        feed_dict = {model.support_item: batch_kshot_item, model.target_user: batch_oracle_user_ebd,
                     model.training_phrase_user_task: is_training}
        evaluate_loss += sess.run(model.loss_user_task, feed_dict)
    return evaluate_loss / len(valid_batch_index)


def training_batch_user_task_fastgcnnew(batch_index, model, sess, train_data, is_training):
    train_target_user, train_k_shot_item, train_second_order_uesrs, train_third_order_items, train_oracle_user_ebd, train_mask_num_second_order_user, train_mask_num_third_order_item = train_data
    for index in batch_index:
        batch_target_user, batch_kshot_item, batch_2nd_user, batch_3rd_item, batch_oracle_user_ebd, batch_mask_num_2nd_user, batch_mask_num_3rd_item = gfn.split_batch_user(
            train_target_user, train_k_shot_item, train_second_order_uesrs, train_third_order_items,
            train_oracle_user_ebd, train_mask_num_second_order_user, train_mask_num_third_order_item, index)
        feed_dict = {model.support_item: batch_kshot_item,
                     model.target_user: batch_oracle_user_ebd,
                     model.training_phrase_user_task: is_training}
        sess.run([model.loss_user_task, model.optimizer_user_task], feed_dict)


def training_loss_user_task_fastgcnnew(batch_index, model, sess, train_data, is_training):
    train_target_user, train_k_shot_item, train_second_order_uesrs, train_third_order_items, train_oracle_user_ebd, train_mask_num_second_order_user, train_mask_num_third_order_item = train_data
    train_loss = 0.0
    num_batch = int(len(train_target_user) / setting.fastgcn_batch_size)
    for index in batch_index:
        batch_target_user, batch_kshot_item, batch_2nd_user, batch_3rd_item, batch_oracle_user_ebd, batch_mask_num_2nd_user, batch_mask_num_3rd_item = gfn.split_batch_user(
            train_target_user, train_k_shot_item, train_second_order_uesrs, train_third_order_items,
            train_oracle_user_ebd, train_mask_num_second_order_user, train_mask_num_third_order_item, index)
        feed_dict = {model.support_item: batch_kshot_item,
                     model.target_user: batch_oracle_user_ebd,
                     model.training_phrase_user_task: is_training}
        train_loss += sess.run(model.loss_user_task, feed_dict)

    return train_loss / num_batch


'''
item_task
'''


def training_batch_item_task_fastgcnnew(batch_index, model, sess, train_data, is_training):
    train_target_item, train_k_shot_user, train_second_order_items, train_third_order_users, train_oracle_item_ebd, train_mask_num_second_order_item, train_mask_num_third_order_user = train_data
    for index in batch_index:
        batch_target_item, batch_kshot_user, batch_2nd_item, batch_3rd_user, batch_oracle_item_ebd, batch_mask_num_2nd_item, batch_mask_num_3rd_user = gfn.split_batch_item(
            train_target_item, train_k_shot_user, train_second_order_items, train_third_order_users,
            train_oracle_item_ebd, train_mask_num_second_order_item, train_mask_num_third_order_user, index)
        feed_dict = {model.support_user: batch_kshot_user,
                     model.target_item: batch_oracle_item_ebd,
                     model.training_phrase_item_task: is_training}
        sess.run([model.loss_item_task, model.optimizer_item_task], feed_dict)


def training_loss_item_task_fastgcnnew(batch_index, model, sess, train_data, is_training):
    train_target_item, train_k_shot_user, train_second_order_items, train_third_order_users, train_oracle_item_ebd, train_mask_num_second_order_item, train_mask_num_third_order_user = train_data
    train_loss = 0.0
    num_batch = int(len(train_target_item) / setting.fastgcn_batch_size)
    for index in batch_index:
        batch_target_item, batch_kshot_user, batch_2nd_item, batch_3rd_user, batch_oracle_item_ebd, batch_mask_num_2nd_item, batch_mask_num_3rd_user = gfn.split_batch_item(
            train_target_item, train_k_shot_user, train_second_order_items, train_third_order_users,
            train_oracle_item_ebd, train_mask_num_second_order_item, train_mask_num_third_order_user, index)
        feed_dict = {model.support_user: batch_kshot_user,
                     model.target_item: batch_oracle_item_ebd,
                     model.training_phrase_item_task: is_training}
        train_loss += sess.run(model.loss_item_task, feed_dict)

    return train_loss / num_batch


def training_item_task_fastgcnnew(model, sess):
    best_loss = 0
    saver = tf.train.Saver(max_to_keep=3)

    ts_i = gfn.Dataset(setting.oracle_training_file_item_task)
    vs_i = gfn.Dataset(setting.oracle_valid_file_item_task)

    num_users = setting.num_users  
    num_items = setting.num_items  
    all_dict = gfn.id_to_useritemid(num_users, num_items)

    train_batches = ts_i.get_positive_instances_item_task(ts_i, all_dict)
    valid_batches = vs_i.get_positive_instances_item_task(vs_i, all_dict)

    num_batch = int(len(train_batches[0])) // setting.fastgcn_batch_size
    batch_index = range(num_batch)
    valid_num_batch = int(len(valid_batches[0])) // setting.fastgcn_batch_size
    valid_batch_index = range(valid_num_batch)

    for epoch_count in range(setting.user_epoch):
        train_begin = time()
        training_batch_item_task_fastgcnnew(batch_index, model, sess, train_batches, True)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_item_task_fastgcnnew(batch_index, model, sess, train_batches, True)
            loss_time = time() - loss_begin
            eval_begin = time()
            cosine = evaluate_item_task_fastgcnnew(valid_batch_index, model, sess, valid_batches, True)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test cosine value is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, cosine))
            if cosine < best_loss:
                best_loss = cosine
                saver.save(sess, setting.checkpoint_path_item_task, global_step=epoch_count)

        ts_i = gfn.Dataset(setting.oracle_training_file_item_task)

        train_batches = ts_i.get_positive_instances_item_task(ts_i, all_dict)


def evaluate_item_task_fastgcnnew(valid_batch_index, model, sess, valid_data, is_training):

    valid_target_item, valid_k_shot_user, valid_second_order_items, valid_third_order_users, valid_oracle_item_ebd, valid_mask_num_second_order_item, valid_mask_num_third_order_user = valid_data
    evaluate_loss = 0.0
    for index in tqdm.tqdm(valid_batch_index):
        batch_target_item, batch_kshot_user, batch_2nd_item, batch_3rd_user, batch_oracle_item_ebd, batch_mask_num_2nd_item, batch_mask_num_3rd_user = gfn.split_batch_item(
            valid_target_item, valid_k_shot_user, valid_second_order_items, valid_third_order_users,
            valid_oracle_item_ebd, valid_mask_num_second_order_item, valid_mask_num_third_order_user, index)

        feed_dict = {model.support_user: batch_kshot_user, model.target_item: batch_oracle_item_ebd,
                     model.training_phrase_item_task: is_training}
        evaluate_loss += sess.run(model.loss_item_task, feed_dict)
    return evaluate_loss / len(valid_batch_index)


'''
2nd user_task, use first-order item and second-order user to predict the target user
'''


def training_2nd_user_task_fastgcnnew(model, sess):
    best_loss = 0
    saver = tf.train.Saver()


    num_users = setting.num_users  
    num_items = setting.num_items  
    all_dict = gfn.id_to_useritemid(num_users, num_items)

    ts_u = gfn.Dataset(setting.oracle_training_file_user_task)

    vs_u = gfn.Dataset(setting.oracle_valid_file_user_task)

    train_batches = ts_u.get_positive_instances_user_task(ts_u, all_dict)
    valid_batches = vs_u.get_positive_instances_user_task(vs_u, all_dict)

    num_batch = int(len(train_batches[0])) // setting.fastgcn_batch_size
    batch_index = range(num_batch)
    valid_num_batch = int(len(valid_batches[0])) // setting.fastgcn_batch_size
    valid_batch_index = range(valid_num_batch)

    for epoch_count in range(setting.second_user_epoch):
        train_begin = time()
        training_batch_2nd_user_task_fastgcnnew(batch_index, model, sess, train_batches, True)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_2nd_user_task_fastgcnnew(batch_index, model, sess, train_batches, True)
            loss_time = time() - loss_begin
            eval_begin = time()
            cosine, pearson = evaluate_2nd_user_task_fastgcnnew(valid_batch_index, model, sess, valid_batches, False)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test cosine value is %.4f, test pearson value is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, cosine, pearson))
            if cosine < best_loss:
                best_loss = cosine
                saver.save(sess, setting.checkpoint_path_user_task, global_step=epoch_count)


        ts_u = gfn.Dataset(setting.oracle_training_file_user_task)
        train_batches = ts_u.get_positive_instances_user_task(ts_u, all_dict)



def training_batch_2nd_user_task_fastgcnnew(batch_index, model, sess, train_data, is_training):
    train_target_user, train_k_shot_item, train_second_order_uesrs, train_third_order_items, train_oracle_user_ebd, train_mask_num_second_order_user, train_mask_num_third_order_item = train_data
    for index in batch_index:
        batch_target_user, batch_kshot_item, batch_2nd_user, batch_3rd_item, batch_oracle_user_ebd, batch_mask_num_2nd_user, batch_mask_num_3rd_item = gfn.split_batch_user(
            train_target_user, train_k_shot_item, train_second_order_uesrs, train_third_order_items,
            train_oracle_user_ebd, train_mask_num_second_order_user, train_mask_num_third_order_item, index)

        feed_dict = {model.target_user: batch_oracle_user_ebd, model.support_item_1st: batch_kshot_item,
                     model.training_phrase_user_task: is_training, model.support_user_2nd: batch_2nd_user,
                     model.training_phrase_item_task: is_training}

        sess.run([model.loss_2nd_user, model.optimizer_2nd_user_task], feed_dict)


def training_loss_2nd_user_task_fastgcnnew(batch_index, model, sess, train_data, is_training):
    train_target_user, train_k_shot_item, train_second_order_uesrs, train_third_order_items, train_oracle_user_ebd, train_mask_num_second_order_user, train_mask_num_third_order_item = train_data
    train_loss = 0.0
    for index in batch_index:
        batch_target_user, batch_kshot_item, batch_2nd_user, batch_3rd_item, batch_oracle_user_ebd, batch_mask_num_2nd_user, batch_mask_num_3rd_item = gfn.split_batch_user(
            train_target_user, train_k_shot_item, train_second_order_uesrs, train_third_order_items,
            train_oracle_user_ebd, train_mask_num_second_order_user, train_mask_num_third_order_item, index)
        feed_dict = {model.target_user: batch_oracle_user_ebd, model.support_item_1st: batch_kshot_item,
                     model.training_phrase_user_task: is_training, model.support_user_2nd: batch_2nd_user,
                     model.training_phrase_item_task: is_training}
        train_loss += sess.run(model.loss_2nd_user, feed_dict)

    return train_loss / len(batch_index)


def evaluate_2nd_user_task_fastgcnnew(valid_batch_index, model, sess, valid_data, is_training):
    valid_target_user, valid_k_shot_item, valid_second_order_uesrs, valid_third_order_items, valid_oracle_user_ebd, valid_mask_num_second_order_user, valid_mask_num_third_order_item = valid_data
    evaluate_loss, evaluate_pearson = 0.0, 0.0
    for index in tqdm.tqdm(valid_batch_index):
        batch_target_user, batch_kshot_item, batch_2nd_user, batch_3rd_item, batch_oracle_user_ebd, batch_mask_num_2nd_user, batch_mask_num_3rd_item = gfn.split_batch_user(
            valid_target_user, valid_k_shot_item, valid_second_order_uesrs, valid_third_order_items,
            valid_oracle_user_ebd, valid_mask_num_second_order_user, valid_mask_num_third_order_item, index)
        feed_dict = {model.target_user: batch_oracle_user_ebd, model.support_item_1st: batch_kshot_item,
                     model.training_phrase_user_task: is_training, model.support_user_2nd: batch_2nd_user,
                     model.training_phrase_item_task: is_training}
        batch_evaluate_loss, batch_predict_ebd, batch_target_ebd = sess.run(
            [model.loss_2nd_user, model.predict_u_2nd, model.target_user], feed_dict)
        evaluate_loss += batch_evaluate_loss

        batch_pearson = Pearson_correlation(batch_predict_ebd, batch_target_ebd)
        evaluate_pearson += batch_pearson
    return evaluate_loss / len(valid_batch_index), evaluate_pearson / len(valid_batch_index)


'''
3rd user_task
'''


def training_3rd_user_task_fastgcnnew(model, sess):
    best_loss = 0
    saver = tf.train.Saver()

    num_users = setting.num_users 
    num_items = setting.num_items  
    all_dict = gfn.id_to_useritemid(num_users, num_items)

    ts_u = gfn.Dataset(setting.oracle_training_file_user_task)

    vs_u = gfn.Dataset(setting.oracle_valid_file_user_task)

    train_batches = ts_u.get_positive_instances_user_task(ts_u, all_dict)
    valid_batches = vs_u.get_positive_instances_user_task(vs_u, all_dict)

    num_batch = int(len(train_batches[0])) // setting.fastgcn_batch_size
    batch_index = range(num_batch)
    valid_num_batch = int(len(valid_batches[0])) // setting.fastgcn_batch_size
    valid_batch_index = range(valid_num_batch)


    for epoch_count in range(setting.third_user_epoch):
        train_begin = time()
        training_batch_3rd_user_task_fastgcnnew(batch_index, model, sess, train_batches, True)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_3rd_user_task_fastgcnnew(batch_index, model, sess, train_batches, True)
            loss_time = time() - loss_begin
            eval_begin = time()
            cosine, pearson = evaluate_3rd_user_task_fastgcnnew(valid_batch_index, model, sess, valid_batches, False)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test cosine value is %.4f, test pearson value is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, cosine, pearson))
            if cosine < best_loss:
                best_loss = cosine
                saver.save(sess, setting.checkpoint_path_user_task, global_step=epoch_count)

        ts_u = gfn.Dataset(setting.oracle_training_file_user_task)

        train_batches = ts_u.get_positive_instances_user_task(ts_u, all_dict)



def training_batch_3rd_user_task_fastgcnnew(batch_index, model, sess, train_data, is_training):
    train_target_user, train_k_shot_item, train_second_order_uesrs, train_third_order_items, train_oracle_user_ebd, train_mask_num_second_order_user, train_mask_num_third_order_item = train_data
    for index in batch_index:
        batch_target_user, batch_kshot_item, batch_2nd_user, batch_3rd_item, batch_oracle_user_ebd, batch_mask_num_2nd_user, batch_mask_num_3rd_item = gfn.split_batch_user(
            train_target_user, train_k_shot_item, train_second_order_uesrs, train_third_order_items,
            train_oracle_user_ebd, train_mask_num_second_order_user, train_mask_num_third_order_item, index)
        feed_dict = {model.target_user: batch_oracle_user_ebd, model.support_item_1st_: batch_kshot_item,
                     model.training_phrase_user_task: is_training, model.support_user_2nd_: batch_2nd_user,
                     model.training_phrase_item_task: is_training,
                     model.support_item_3rd: batch_3rd_item}

        sess.run([model.loss_3rd_user, model.optimizer_3rd_user_task], feed_dict)


def training_loss_3rd_user_task_fastgcnnew(batch_index, model, sess, train_data, is_training):
    train_target_user, train_k_shot_item, train_second_order_uesrs, train_third_order_items, train_oracle_user_ebd, train_mask_num_second_order_user, train_mask_num_third_order_item = train_data
    train_loss = 0.0
    for index in batch_index:
        batch_target_user, batch_kshot_item, batch_2nd_user, batch_3rd_item, batch_oracle_user_ebd, batch_mask_num_2nd_user, batch_mask_num_3rd_item = gfn.split_batch_user(
            train_target_user, train_k_shot_item, train_second_order_uesrs, train_third_order_items,
            train_oracle_user_ebd, train_mask_num_second_order_user, train_mask_num_third_order_item, index)
        feed_dict = {model.target_user: batch_oracle_user_ebd, model.support_item_1st_: batch_kshot_item,
                     model.training_phrase_user_task: is_training, model.support_user_2nd_: batch_2nd_user,
                     model.training_phrase_item_task: is_training,
                     model.support_item_3rd: batch_3rd_item}
        train_loss += sess.run(model.loss_3rd_user, feed_dict)

    return train_loss / len(batch_index)


def evaluate_3rd_user_task_fastgcnnew(valid_batch_index, model, sess, valid_data, is_training):
    valid_target_user, valid_k_shot_item, valid_second_order_uesrs, valid_third_order_items, valid_oracle_user_ebd, valid_mask_num_second_order_user, valid_mask_num_third_order_item = valid_data
    evaluate_loss, evaluate_pearson = 0.0, 0.0
    for index in tqdm.tqdm(valid_batch_index):
        batch_target_user, batch_kshot_item, batch_2nd_user, batch_3rd_item, batch_oracle_user_ebd, batch_mask_num_2nd_user, batch_mask_num_3rd_item = gfn.split_batch_user(
            valid_target_user, valid_k_shot_item, valid_second_order_uesrs, valid_third_order_items,
            valid_oracle_user_ebd, valid_mask_num_second_order_user, valid_mask_num_third_order_item, index)
        feed_dict = {model.target_user: batch_oracle_user_ebd, model.support_item_1st_: batch_kshot_item,
                     model.training_phrase_user_task: is_training, model.support_user_2nd_: batch_2nd_user,
                     model.training_phrase_item_task: is_training,
                     model.support_item_3rd: batch_3rd_item}
        batch_evaluate_loss, batch_predict_ebd, batch_target_ebd = sess.run(
            [model.loss_3rd_user, model.predict_u_3rd, model.target_user], feed_dict)
        evaluate_loss += batch_evaluate_loss

        batch_pearson = Pearson_correlation(batch_predict_ebd, batch_target_ebd)
        evaluate_pearson += batch_pearson
    return evaluate_loss / len(valid_batch_index), evaluate_pearson / len(valid_batch_index)


###########################################################
# item_task
'''
2nd item_task, use first-order user and second-order item to predict the target item
'''


def training_2nd_item_task_fastgcnnew(model, sess):
    best_loss = 0
    saver = tf.train.Saver()

    num_users = setting.num_users 
    num_items = setting.num_items  
    all_dict = gfn.id_to_useritemid(num_users, num_items)

    ts_i = gfn.Dataset(setting.oracle_training_file_item_task)
    vs_i = gfn.Dataset(setting.oracle_valid_file_item_task)
    train_batches = ts_i.get_positive_instances_item_task(ts_i, all_dict)
    valid_batches = vs_i.get_positive_instances_item_task(vs_i, all_dict)

    num_batch = int(len(train_batches[0]) // setting.fastgcn_batch_size )
    batch_index = range(num_batch)
    valid_num_batch = int(len(valid_batches[0]) // setting.fastgcn_batch_size)
    valid_batch_index = range(valid_num_batch)

    for epoch_count in range(setting.second_item_epoch):
        train_begin = time()
        training_batch_2nd_item_task_fastgcnnew(batch_index, model, sess, train_batches, True)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_2nd_item_task_fastgcnnew(batch_index, model, sess, train_batches, True)
            loss_time = time() - loss_begin
            eval_begin = time()
            cosine, pearson = evaluate_2nd_item_task_fastgcnnew(valid_batch_index, model, sess, valid_batches, False)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test cosine value is %.4f, test pearson value is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, cosine, pearson))
            if cosine < best_loss:
                best_loss = cosine
                saver.save(sess, setting.checkpoint_path_item_task, global_step=epoch_count)

        ts_i = gfn.Dataset(setting.oracle_training_file_item_task)
        train_batches = ts_i.get_positive_instances_item_task(ts_i, all_dict)


def training_batch_2nd_item_task_fastgcnnew(batch_index, model, sess, train_data, is_training):
    train_target_item, train_k_shot_user, train_second_order_items, train_third_order_users, train_oracle_item_ebd, train_mask_num_second_order_item, train_mask_num_third_order_user = train_data
    for index in batch_index:
        batch_target_item, batch_kshot_user, batch_2nd_item, batch_3rd_user, batch_oracle_item_ebd, batch_mask_num_2nd_item, batch_mask_num_3rd_user = gfn.split_batch_item(
            train_target_item, train_k_shot_user, train_second_order_items, train_third_order_users,
            train_oracle_item_ebd, train_mask_num_second_order_item, train_mask_num_third_order_user, index)

        feed_dict = {model.target_item: batch_oracle_item_ebd, model.support_user_1st_pos: batch_kshot_user,
                     model.training_phrase_user_task: is_training, model.support_item_2nd_pos: batch_2nd_item,
                     model.training_phrase_item_task: is_training}

        sess.run([model.loss_2nd_item_pos, model.optimizer_2nd_item_task_pos], feed_dict)


def training_loss_2nd_item_task_fastgcnnew(batch_index, model, sess, train_data, is_training):
    train_loss = 0.0
    train_target_item, train_k_shot_user, train_second_order_items, train_third_order_users, train_oracle_item_ebd, train_mask_num_second_order_item, train_mask_num_third_order_user = train_data
    for index in batch_index:
        batch_target_item, batch_kshot_user, batch_2nd_item, batch_3rd_user, batch_oracle_item_ebd, batch_mask_num_2nd_item, batch_mask_num_3rd_user = gfn.split_batch_item(
            train_target_item, train_k_shot_user, train_second_order_items, train_third_order_users,
            train_oracle_item_ebd, train_mask_num_second_order_item, train_mask_num_third_order_user, index)

        feed_dict = {model.target_item: batch_oracle_item_ebd, model.support_user_1st_pos: batch_kshot_user,
                     model.training_phrase_user_task: is_training, model.support_item_2nd_pos: batch_2nd_item,
                     model.training_phrase_item_task: is_training}
        train_loss += sess.run(model.loss_2nd_item_pos, feed_dict)

    return train_loss / len(batch_index)


def evaluate_2nd_item_task_fastgcnnew(valid_batch_index, model, sess, valid_data, is_training):
    evaluate_loss, evaluate_pearson = 0.0, 0.0
    valid_target_item, valid_k_shot_user, valid_second_order_items, valid_third_order_users, valid_oracle_item_ebd, valid_mask_num_second_order_item, valid_mask_num_third_order_user = valid_data
    for index in tqdm.tqdm(valid_batch_index):
        batch_target_item, batch_kshot_user, batch_2nd_item, batch_3rd_user, batch_oracle_item_ebd, batch_mask_num_2nd_item, batch_mask_num_3rd_user = gfn.split_batch_item(
            valid_target_item, valid_k_shot_user, valid_second_order_items, valid_third_order_users,
            valid_oracle_item_ebd, valid_mask_num_second_order_item, valid_mask_num_third_order_user, index)
        feed_dict = {model.target_item: batch_oracle_item_ebd, model.support_user_1st_pos: batch_kshot_user,
                     model.training_phrase_user_task: is_training, model.support_item_2nd_pos: batch_2nd_item,
                     model.training_phrase_item_task: is_training}
        batch_evaluate_loss, batch_predict_ebd, batch_target_ebd = sess.run(
            [model.loss_2nd_item_pos, model.predict_i_2nd_pos, model.target_item], feed_dict)
        evaluate_loss += batch_evaluate_loss

        batch_pearson = Pearson_correlation(batch_predict_ebd, batch_target_ebd)
        evaluate_pearson += batch_pearson
    return evaluate_loss / len(valid_batch_index), evaluate_pearson / len(valid_batch_index)


'''
3rd item_task
'''


def training_3rd_item_task_fastgcnnew(model, sess):
    best_loss = 0
    saver = tf.train.Saver()

   
    num_users = setting.num_users  
    num_items = setting.num_items  
    all_dict = gfn.id_to_useritemid(num_users, num_items)

    ts_i = gfn.Dataset(setting.oracle_training_file_item_task)
    vs_i = gfn.Dataset(setting.oracle_valid_file_item_task)
    train_batches = ts_i.get_positive_instances_item_task(ts_i, all_dict)
    valid_batches = vs_i.get_positive_instances_item_task(vs_i, all_dict)

    num_batch = int(len(train_batches[0])) // setting.fastgcn_batch_size
    batch_index = range(num_batch)
    valid_num_batch = int(len(valid_batches[0])) // setting.fastgcn_batch_size
    valid_batch_index = range(valid_num_batch)

    for epoch_count in range(setting.third_item_epoch):
        train_begin = time()
        training_batch_3rd_item_task_fastgcnnew(batch_index, model, sess, train_batches, True)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_3rd_item_task_fastgcnnew(batch_index, model, sess, train_batches, True)
            loss_time = time() - loss_begin
            eval_begin = time()
            cosine, pearson = evaluate_3rd_item_task_fastgcnnew(valid_batch_index, model, sess, valid_batches, False)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test cosine value is %.4f, test pearson value is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, cosine, pearson))
            if cosine < best_loss:
                best_loss = cosine
                saver.save(sess, setting.checkpoint_path_item_task, global_step=epoch_count)

        ts_i = gfn.Dataset(setting.oracle_training_file_item_task)
        train_batches = ts_i.get_positive_instances_item_task(ts_i, all_dict)



def training_batch_3rd_item_task_fastgcnnew(batch_index, model, sess, train_data, is_training):
    train_target_item, train_k_shot_user, train_second_order_items, train_third_order_users, train_oracle_item_ebd, train_mask_num_second_order_item, train_mask_num_third_order_user = train_data
    for index in batch_index:
        batch_target_item, batch_kshot_user, batch_2nd_item, batch_3rd_user, batch_oracle_item_ebd, batch_mask_num_2nd_item, batch_mask_num_3rd_user = gfn.split_batch_item(
            train_target_item, train_k_shot_user, train_second_order_items, train_third_order_users,
            train_oracle_item_ebd, train_mask_num_second_order_item, train_mask_num_third_order_user, index)
        feed_dict = {model.target_item: batch_oracle_item_ebd, model.support_user_1st_pos_: batch_kshot_user,
                     model.training_phrase_user_task: is_training, model.support_item_2nd_pos_: batch_2nd_item,
                     model.training_phrase_item_task: is_training,
                     model.support_user_3rd_pos: batch_3rd_user}

        sess.run([model.loss_3rd_item_pos, model.optimizer_3rd_item_task_pos], feed_dict)


def training_loss_3rd_item_task_fastgcnnew(batch_index, model, sess, train_data, is_training):
    train_loss = 0.0
    train_target_item, train_k_shot_user, train_second_order_items, train_third_order_users, train_oracle_item_ebd, train_mask_num_second_order_item, train_mask_num_third_order_user = train_data
    for index in batch_index:
        batch_target_item, batch_kshot_user, batch_2nd_item, batch_3rd_user, batch_oracle_item_ebd, batch_mask_num_2nd_item, batch_mask_num_3rd_user = gfn.split_batch_item(
            train_target_item, train_k_shot_user, train_second_order_items, train_third_order_users,
            train_oracle_item_ebd, train_mask_num_second_order_item, train_mask_num_third_order_user, index)

        feed_dict = {model.target_item: batch_oracle_item_ebd, model.support_user_1st_pos_: batch_kshot_user,
                     model.training_phrase_user_task: is_training, model.support_item_2nd_pos_: batch_2nd_item,
                     model.training_phrase_item_task: is_training,
                     model.support_user_3rd_pos: batch_3rd_user}
        train_loss += sess.run(model.loss_3rd_item_pos, feed_dict)

    return train_loss / len(batch_index)


def evaluate_3rd_item_task_fastgcnnew(valid_batch_index, model, sess, valid_data, is_training):
    evaluate_loss, evaluate_pearson = 0.0, 0.0
    valid_target_item, valid_k_shot_user, valid_second_order_items, valid_third_order_users, valid_oracle_item_ebd, valid_mask_num_second_order_item, valid_mask_num_third_order_user = valid_data
    for index in tqdm.tqdm(valid_batch_index):
        batch_target_item, batch_kshot_user, batch_2nd_item, batch_3rd_user, batch_oracle_item_ebd, batch_mask_num_2nd_item, batch_mask_num_3rd_user = gfn.split_batch_item(
            valid_target_item, valid_k_shot_user, valid_second_order_items, valid_third_order_users,
            valid_oracle_item_ebd, valid_mask_num_second_order_item, valid_mask_num_third_order_user, index)

        feed_dict = {model.target_item: batch_oracle_item_ebd, model.support_user_1st_pos_: batch_kshot_user,
                     model.training_phrase_user_task: is_training, model.support_item_2nd_pos_: batch_2nd_item,
                     model.training_phrase_item_task: is_training,
                     model.support_user_3rd_pos: batch_3rd_user}
        batch_evaluate_loss, batch_predict_ebd, batch_target_ebd = sess.run(
            [model.loss_3rd_item_pos, model.predict_i_3rd_pos, model.target_item], feed_dict)
        evaluate_loss += batch_evaluate_loss

        batch_pearson = Pearson_correlation(batch_predict_ebd, batch_target_ebd)
        evaluate_pearson += batch_pearson
    return evaluate_loss / len(valid_batch_index), evaluate_pearson / len(valid_batch_index)

