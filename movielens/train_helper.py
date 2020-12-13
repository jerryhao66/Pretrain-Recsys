__author__ = 'haobowen'
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import logging
import numpy as np
import setting
import gendata_may as data
from time import time
import tqdm
from scipy.stats import pearsonr
from unit import *


'''
uesr_task
'''


def training_user_task(model, sess):
    best_loss = 0
    saver = tf.train.Saver()
    ts_u = data.generate_user_dict_train(setting.oracle_training_file_user_task)

    vs_u = data.generate_user_dict_valid(setting.oracle_valid_file_user_task)
    train_batches = data.generate_meta_train_user_set(ts_u)
    valid_batches = data.generate_meta_valid_user_set(vs_u)

    num_batch = int(train_batches[3]) // setting.batch_size
    batch_index = range(num_batch)
    valid_num_batch = int(valid_batches[3]) // setting.batch_size
    valid_batch_index = range(valid_num_batch)

    for epoch_count in range(setting.user_epoch):
        train_begin = time()
        training_batch_user_task(batch_index, model, sess, train_batches, True)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_user_task(batch_index, model, sess, train_batches, True)
            loss_time = time() - loss_begin
            eval_begin = time()
            cosine = evaluate_user_task(valid_batch_index, model, sess, valid_batches, True)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test cosine value is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, cosine))

            if cosine < best_loss:
                best_loss = cosine
                saver.save(sess, setting.checkpoint_path_user_task, global_step=epoch_count)

        ts_u = data.generate_user_dict_train(setting.oracle_training_file_user_task)

        vs_u = data.generate_user_dict_valid(setting.oracle_valid_file_user_task)
        train_batches = data.generate_meta_train_user_set(ts_u)
        valid_batches = data.generate_meta_valid_user_set(vs_u)


def evaluate_user_task(valid_batch_index, model, sess, valid_data, is_training):
    '''
    the train_batch_size not necessarily equal to the test_batch_size
    '''
    evaluate_loss = 0.0
    for index in tqdm.tqdm(valid_batch_index):
        user_id, support_item, target_user = data.batch_gen_user_task(valid_data, index, setting.batch_size)

        feed_dict = {model.support_item: support_item, model.target_user: target_user,
                     model.training_phrase_user_task: is_training}
        evaluate_loss += sess.run(model.loss_user_task, feed_dict)
    return evaluate_loss / len(valid_batch_index)


def training_batch_user_task(batch_index, model, sess, train_data, is_training):
    for index in batch_index:
        _, support_item, target_user = data.batch_gen_user_task(train_data, index, setting.batch_size)
        feed_dict = {model.support_item: support_item,
                     model.target_user: target_user,
                     model.training_phrase_user_task: is_training}
        sess.run([model.loss_user_task, model.optimizer_user_task], feed_dict)


def training_loss_user_task(batch_index, model, sess, train_data, is_training):
    train_loss = 0.0
    num_batch = int(train_data[3] / setting.batch_size)
    for index in batch_index:
        _, support_item, target_user = data.batch_gen_user_task(train_data, index, setting.batch_size)
        feed_dict = {model.support_item: support_item,
                     model.target_user: target_user,
                     model.training_phrase_user_task: is_training}
        train_loss += sess.run(model.loss_user_task, feed_dict)

    return train_loss / num_batch


'''
item_task
'''


def training_batch_item_task(batch_index, model, sess, train_data, is_training):
    for index in batch_index:
        _, support_user, target_item = data.batch_gen_item_task(train_data, index, setting.batch_size)
        feed_dict = {model.support_user: support_user,
                     model.target_item: target_item,
                     model.training_phrase_item_task: is_training}
        sess.run([model.loss_item_task, model.optimizer_item_task], feed_dict)


def training_loss_item_task(batch_index, model, sess, train_data, is_training):
    train_loss = 0.0
    num_batch = int(train_data[3] / setting.batch_size)
    for index in batch_index:
        _, support_user, target_item = data.batch_gen_item_task(train_data, index, setting.batch_size)
        feed_dict = {model.support_user: support_user,
                     model.target_item: target_item,
                     model.training_phrase_item_task: is_training}
        train_loss += sess.run(model.loss_item_task, feed_dict)

    return train_loss / num_batch


def training_item_task(model, sess):
    best_loss = 0
    saver = tf.train.Saver(max_to_keep=3)
    ts_i = data.generate_item_dict_train(setting.oracle_training_file_item_task)
    vs_i = data.generate_item_dict_valid(setting.oracle_valid_file_item_task)

    train_batches = data.generate_meta_train_item_set(ts_i)
    valid_batches = data.generate_meta_valid_item_set(vs_i)

    num_batch = int(train_batches[3]) // setting.batch_size
    batch_index = range(num_batch)
    valid_num_batch = int(valid_batches[3]) // setting.batch_size
    valid_batch_index = range(valid_num_batch)

    for epoch_count in range(setting.user_epoch):
        train_begin = time()
        training_batch_item_task(batch_index, model, sess, train_batches, True)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_item_task(batch_index, model, sess, train_batches, True)
            loss_time = time() - loss_begin
            eval_begin = time()
            cosine = evaluate_item_task(valid_batch_index, model, sess, valid_batches, True)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test cosine value is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, cosine))
            if cosine < best_loss:
                best_loss = cosine
                saver.save(sess, setting.checkpoint_path_item_task, global_step=epoch_count)

        ts_i = data.generate_item_dict_train(setting.oracle_training_file_item_task)
        vs_i = data.generate_item_dict_valid(setting.oracle_valid_file_item_task)

        train_batches = data.generate_meta_train_item_set(ts_i)
        valid_batches = data.generate_meta_valid_item_set(vs_i)


def evaluate_item_task(valid_batch_index, model, sess, valid_data, is_training):

    evaluate_loss = 0.0
    for index in tqdm.tqdm(valid_batch_index):
        item_id, support_user, target_item = data.batch_gen_item_task(valid_data, index, setting.batch_size)

        feed_dict = {model.support_user: support_user, model.target_item: target_item,
                     model.training_phrase_item_task: is_training}
        evaluate_loss += sess.run(model.loss_item_task, feed_dict)
    return evaluate_loss / len(valid_batch_index)


'''
2nd user_task, use first-order item and second-order user to predict the target user
'''


def training_2nd_user_task(model, sess):
    best_loss = 0
    saver = tf.train.Saver()

    data_train = data.Dataset(setting.oracle_training_file_user_task)
    train_batches = data_train.get_positive_instances_user_task(random_seed=0)
    num_batch_train = data_train.oracle_num_users // setting.batch_size + 1
    train_batch_index = range(num_batch_train)

    data_valid = data.Dataset(setting.oracle_valid_file_user_task)
    valid_batches = data_valid.get_positive_instances_user_task(random_seed=0)
    num_batch_valid = data_valid.oracle_num_users // setting.batch_size + 1
    valid_batch_index = range(num_batch_valid)

    for epoch_count in range(setting.second_user_epoch):
        train_begin = time()
        training_batch_2nd_user_task(data_train, train_batch_index, model, sess, train_batches, True)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_2nd_user_task(data_train, train_batch_index, model, sess, train_batches, True)
            loss_time = time() - loss_begin
            eval_begin = time()
            cosine, pearson = evaluate_2nd_user_task(data_valid, valid_batch_index, model, sess, valid_batches, False)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test cosine value is %.4f, test pearson value is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, cosine, pearson))
            if cosine < best_loss:
                best_loss = cosine
                saver.save(sess, setting.checkpoint_path_user_task, global_step=epoch_count)

        data_train = data.Dataset(setting.oracle_training_file_user_task)
        train_batches = data_train.get_positive_instances_user_task(random_seed=epoch_count + 1)
        num_batch_train = data_train.oracle_num_users // setting.batch_size + 1
        train_batch_index = range(num_batch_train)


def training_batch_2nd_user_task(data, batch_index, model, sess, train_data, is_training):
    for index in batch_index:
        b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item = data.batch_gen_3rd_user_task(
            train_data, index)

        feed_dict = {model.target_user: b_oracle_user_ebd, model.support_item_1st: b_k_shot_item,
                     model.training_phrase_user_task: is_training, model.support_user_2nd: b_second_order_users,
                     model.training_phrase_item_task: is_training}

        sess.run([model.loss_2nd_user, model.optimizer_2nd_user_task], feed_dict)


def training_loss_2nd_user_task(data, batch_index, model, sess, train_data, is_training):
    train_loss = 0.0
    num_batch = data.oracle_num_users // setting.batch_size
    for index in batch_index:
        b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item = data.batch_gen_3rd_user_task(
            train_data, index)
        feed_dict = {model.target_user: b_oracle_user_ebd, model.support_item_1st: b_k_shot_item,
                     model.training_phrase_user_task: is_training, model.support_user_2nd: b_second_order_users,
                     model.training_phrase_item_task: is_training}
        train_loss += sess.run(model.loss_2nd_user, feed_dict)

    return train_loss / num_batch


def evaluate_2nd_user_task(data, valid_batch_index, model, sess, valid_data, is_training):
    evaluate_loss, evaluate_pearson = 0.0, 0.0
    for index in tqdm.tqdm(valid_batch_index):
        b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item = data.batch_gen_3rd_user_task(
            valid_data, index)
        feed_dict = {model.target_user: b_oracle_user_ebd, model.support_item_1st: b_k_shot_item,
                     model.training_phrase_user_task: is_training, model.support_user_2nd: b_second_order_users,
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


def training_3rd_user_task(model, sess):
    best_loss = 0
    saver = tf.train.Saver()



    data_train = data.Dataset(setting.oracle_training_file_user_task)
    train_batches = data_train.get_positive_instances_user_task(random_seed=0)
    num_batch_train = data_train.oracle_num_users // setting.batch_size + 1
    train_batch_index = range(num_batch_train)

    data_valid = data.Dataset(setting.oracle_valid_file_user_task)
    valid_batches = data_valid.get_positive_instances_user_task(random_seed=0)
    num_batch_valid = data_valid.oracle_num_users // setting.batch_size + 1
    valid_batch_index = range(num_batch_valid)

    for epoch_count in range(setting.third_user_epoch):
        train_begin = time()
        training_batch_3rd_user_task(data_train, train_batch_index, model, sess, train_batches, True)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_3rd_user_task(data_train, train_batch_index, model, sess, train_batches, True)
            loss_time = time() - loss_begin
            eval_begin = time()
            cosine, pearson = evaluate_3rd_user_task(data_valid, valid_batch_index, model, sess, valid_batches, False)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test cosine value is %.4f, test pearson value is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, cosine, pearson))
            if cosine < best_loss:
                best_loss = cosine
                saver.save(sess, setting.checkpoint_path_user_task, global_step=epoch_count)

        data_train = data.Dataset(setting.oracle_training_file_user_task)
        train_batches = data_train.get_positive_instances_user_task(random_seed=epoch_count + 1)
        num_batch_train = data_train.oracle_num_users // setting.batch_size + 1
        train_batch_index = range(num_batch_train)


def training_batch_3rd_user_task(data, batch_index, model, sess, train_data, is_training):
    for index in batch_index:
        b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item = data.batch_gen_3rd_user_task(
            train_data, index)
        feed_dict = {model.target_user: b_oracle_user_ebd, model.support_item_1st_: b_k_shot_item,
                     model.training_phrase_user_task: is_training, model.support_user_2nd_: b_second_order_users,
                     model.training_phrase_item_task: is_training,
                     model.support_item_3rd: b_third_order_items}

        sess.run([model.loss_3rd_user, model.optimizer_3rd_user_task], feed_dict)


def training_loss_3rd_user_task(data, batch_index, model, sess, train_data, is_training):
    train_loss = 0.0
    num_batch = data.oracle_num_users // setting.batch_size
    for index in batch_index:
        b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item = data.batch_gen_3rd_user_task(
            train_data, index)
        feed_dict = {model.target_user: b_oracle_user_ebd, model.support_item_1st_: b_k_shot_item,
                     model.training_phrase_user_task: is_training, model.support_user_2nd_: b_second_order_users,
                     model.training_phrase_item_task: is_training,
                     model.support_item_3rd: b_third_order_items}
        train_loss += sess.run(model.loss_3rd_user, feed_dict)

    return train_loss / num_batch


def evaluate_3rd_user_task(data, valid_batch_index, model, sess, valid_data, is_training):
    evaluate_loss, evaluate_pearson = 0.0, 0.0
    for index in tqdm.tqdm(valid_batch_index):
        b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item = data.batch_gen_3rd_user_task(
            valid_data, index)
        feed_dict = {model.target_user: b_oracle_user_ebd, model.support_item_1st_: b_k_shot_item,
                     model.training_phrase_user_task: is_training, model.support_user_2nd_: b_second_order_users,
                     model.training_phrase_item_task: is_training,
                     model.support_item_3rd: b_third_order_items}
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


def training_2nd_item_task(model, sess):
    best_loss = 0
    saver = tf.train.Saver()


    data_train = data.Dataset(setting.oracle_training_file_item_task)
    train_batches = data_train.get_positive_instances_item_task(random_seed=0)
    num_batch_train = data_train.oracle_num_items // setting.batch_size + 1
    train_batch_index = range(num_batch_train)

    data_valid = data.Dataset(setting.oracle_valid_file_item_task)
    valid_batches = data_valid.get_positive_instances_item_task(random_seed=0)
    num_batch_valid = data_valid.oracle_num_items // setting.batch_size + 1
    valid_batch_index = range(num_batch_valid)

    for epoch_count in range(setting.second_item_epoch):
        train_begin = time()
        training_batch_2nd_item_task(data_train, train_batch_index, model, sess, train_batches, True)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_2nd_item_task(data_train, train_batch_index, model, sess, train_batches, True)
            loss_time = time() - loss_begin
            eval_begin = time()
            cosine, pearson = evaluate_2nd_item_task(data_valid, valid_batch_index, model, sess, valid_batches, False)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test cosine value is %.4f, test pearson value is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, cosine, pearson))
            if cosine < best_loss:
                best_loss = cosine
                saver.save(sess, setting.checkpoint_path_item_task, global_step=epoch_count)

        data_train = data.Dataset(setting.oracle_training_file_item_task)
        train_batches = data_train.get_positive_instances_item_task(random_seed=epoch_count + 1)
        num_batch_train = data_train.oracle_num_items // setting.batch_size + 1
        train_batch_index = range(num_batch_train)


def training_batch_2nd_item_task(data, batch_index, model, sess, train_data, is_training):
    for index in batch_index:
        b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user = data.batch_gen_3rd_item_task(
            train_data, index)
        feed_dict = {model.target_item: b_oracle_item_ebd, model.support_user_1st_pos: b_k_shot_user,
                     model.training_phrase_user_task: is_training, model.support_item_2nd_pos: b_second_order_items,
                     model.training_phrase_item_task: is_training}

        sess.run([model.loss_2nd_item_pos, model.optimizer_2nd_item_task_pos], feed_dict)


def training_loss_2nd_item_task(data, batch_index, model, sess, train_data, is_training):
    train_loss = 0.0
    num_batch = data.oracle_num_items // setting.batch_size
    for index in batch_index:
        b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user = data.batch_gen_3rd_item_task(
            train_data, index)
        feed_dict = {model.target_item: b_oracle_item_ebd, model.support_user_1st_pos: b_k_shot_user,
                     model.training_phrase_user_task: is_training, model.support_item_2nd_pos: b_second_order_items,
                     model.training_phrase_item_task: is_training}
        train_loss += sess.run(model.loss_2nd_item_pos, feed_dict)

    return train_loss / num_batch


def evaluate_2nd_item_task(data, valid_batch_index, model, sess, valid_data, is_training):
    evaluate_loss, evaluate_pearson = 0.0, 0.0
    for index in tqdm.tqdm(valid_batch_index):
        b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user = data.batch_gen_3rd_item_task(
            valid_data, index)
        feed_dict = {model.target_item: b_oracle_item_ebd, model.support_user_1st_pos: b_k_shot_user,
                     model.training_phrase_user_task: is_training, model.support_item_2nd_pos: b_second_order_items,
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


def training_3rd_item_task(model, sess):
    best_loss = 0
    saver = tf.train.Saver()


    data_train = data.Dataset(setting.oracle_training_file_item_task)
    train_batches = data_train.get_positive_instances_item_task(random_seed=0)
    num_batch_train = data_train.oracle_num_items // setting.batch_size + 1
    train_batch_index = range(num_batch_train)

    data_valid = data.Dataset(setting.oracle_valid_file_item_task)
    valid_batches = data_valid.get_positive_instances_item_task(random_seed=0)
    num_batch_valid = data_valid.oracle_num_items // setting.batch_size + 1
    valid_batch_index = range(num_batch_valid)

    for epoch_count in range(setting.third_item_epoch):
        train_begin = time()
        training_batch_3rd_item_task(data_train, train_batch_index, model, sess, train_batches, True)
        train_time = time() - train_begin
        if epoch_count % setting.verbose == 0:
            loss_begin = time()
            train_loss = training_loss_3rd_item_task(data_train, train_batch_index, model, sess, train_batches, True)
            loss_time = time() - loss_begin
            eval_begin = time()
            cosine, pearson = evaluate_3rd_item_task(data_valid, valid_batch_index, model, sess, valid_batches, False)
            eval_time = time() - eval_begin
            print(
                'epoch %d, train time is %.4f, loss time is %.4f, eval_time is %.4f, train_loss is %.4f, test cosine value is %.4f, test pearson value is %.4f' % (
                    epoch_count, train_time, loss_time, eval_time, train_loss, cosine, pearson))
            if cosine < best_loss:
                best_loss = cosine
                saver.save(sess, setting.checkpoint_path_item_task, global_step=epoch_count)

        data_train = data.Dataset(setting.oracle_training_file_item_task)
        train_batches = data_train.get_positive_instances_item_task(random_seed=epoch_count + 1)
        num_batch_train = data_train.oracle_num_items // setting.batch_size + 1
        train_batch_index = range(num_batch_train)


def training_batch_3rd_item_task(data, batch_index, model, sess, train_data, is_training):
    for index in batch_index:
        b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user = data.batch_gen_3rd_item_task(
            train_data, index)
        feed_dict = {model.target_item: b_oracle_item_ebd, model.support_user_1st_pos_: b_k_shot_user,
                     model.training_phrase_user_task: is_training, model.support_item_2nd_pos_: b_second_order_items,
                     model.training_phrase_item_task: is_training,
                     model.support_user_3rd_pos: b_third_order_users}

        sess.run([model.loss_3rd_item_pos, model.optimizer_3rd_item_task_pos], feed_dict)


def training_loss_3rd_item_task(data, batch_index, model, sess, train_data, is_training):
    train_loss = 0.0
    num_batch = data.oracle_num_items // setting.batch_size
    for index in batch_index:
        b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user = data.batch_gen_3rd_item_task(
            train_data, index)
        feed_dict = {model.target_item: b_oracle_item_ebd, model.support_user_1st_pos_: b_k_shot_user,
                     model.training_phrase_user_task: is_training, model.support_item_2nd_pos_: b_second_order_items,
                     model.training_phrase_item_task: is_training,
                     model.support_user_3rd_pos: b_third_order_users}
        train_loss += sess.run(model.loss_3rd_item_pos, feed_dict)

    return train_loss / num_batch


def evaluate_3rd_item_task(data, valid_batch_index, model, sess, valid_data, is_training):
    evaluate_loss, evaluate_pearson = 0.0, 0.0
    for index in tqdm.tqdm(valid_batch_index):
        b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user = data.batch_gen_3rd_item_task(
            valid_data, index)
        feed_dict = {model.target_item: b_oracle_item_ebd, model.support_user_1st_pos_: b_k_shot_user,
                     model.training_phrase_user_task: is_training, model.support_item_2nd_pos_: b_second_order_items,
                     model.training_phrase_item_task: is_training,
                     model.support_user_3rd_pos: b_third_order_users}
        batch_evaluate_loss, batch_predict_ebd, batch_target_ebd = sess.run(
            [model.loss_3rd_item_pos, model.predict_i_3rd_pos, model.target_item], feed_dict)
        evaluate_loss += batch_evaluate_loss

        batch_pearson = Pearson_correlation(batch_predict_ebd, batch_target_ebd)
        evaluate_pearson += batch_pearson
    return evaluate_loss / len(valid_batch_index), evaluate_pearson / len(valid_batch_index)

