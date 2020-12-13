__author__ = 'haobowen'

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import setting
import numpy as np
from unit import *
from train_helper import *
from train_rl_helper import *
import logging

import tqdm
import gendata_may as data

import tensorflow as tf
from downstream_trainer import *


class GeneralGNN(object):
    def __init__(self, name, sess):
        self.name = name

        self.embedding_size = setting.embedding_size
        self.learning_rate = setting.learning_rate
        self.learning_rate_downstream = setting.learning_rate_downstream
        self.num_items = setting.num_items
        self.num_users = setting.num_users
        self.k = setting.k
        self.dropout = setting.dropout
        self.batch_size = setting.batch_size
        self.decay = setting.decay

        # transformer encoder structure
        self.dropout_rate = setting.dropout_rate
        self.num_heads = setting.num_heads
        self.d_ff = setting.d_ff
        self.num_blocks = setting.num_blocks

        with tf.device('/gpu:0'):
            self._create_placeholder()

            self.final_support_encode_user_task, self.batch_loss_user_task, self.loss_user_task, self.optimizer_user_task = self._1st_user_task()
            self.final_support_encode_item_task, self.batch_loss_item_task, self.loss_item_task, self.optimizer_item_task = self._1st_item_task()

            self.pre_train_i_ax = setting.pre_train_i_ax_path
            self.pre_train_u_ax = setting.pre_train_u_ax_path
            self.pre_train_u_aax = setting.pre_train_u_aax_path
            self.pre_train_i_aax = setting.pre_train_i_aax_path
            self.pre_train_i_aaax = setting.pre_train_i_aaax_path
            self.pre_train_u_aaax = setting.pre_train_u_aaax_path

            self.predict_u_2nd, self.batch_loss_2nd_user, self.loss_2nd_user, self.optimizer_2nd_user_task = self._2nd_user_task(
                name, self.support_user_2nd, self.support_item_1st)

        with tf.device('/gpu:1'):
            self.predict_u_3rd, self.batch_loss_3rd_user, self.loss_3rd_user, self.optimizer_3rd_user_task = self._3rd_user_task(
                name, self.support_item_3rd, self.support_user_2nd_, self.support_item_1st_)

        with tf.device('/gpu:2'):
            self.predict_i_2nd_pos, self.batch_loss_2nd_item_pos, self.loss_2nd_item_pos, self.optimizer_2nd_item_task_pos = self._2nd_item_task(
                name, self.support_item_2nd_pos, self.support_user_1st_pos)
            self.predict_i_2nd_neg, self.batch_loss_2nd_item_neg, self.loss_2nd_item_neg, self.optimizer_2nd_item_task_neg = self._2nd_item_task(
                name, self.support_item_2nd_neg, self.support_user_1st_neg)

        with tf.device('/gpu:3'):
            self.predict_i_3rd_pos, self.batch_loss_3rd_item_pos, self.loss_3rd_item_pos, self.optimizer_3rd_item_task_pos = self._3rd_item_task(
                name, self.support_user_3rd_pos, self.support_item_2nd_pos_, self.support_user_1st_pos_)
        with tf.device('/gpu:0'):
            self.predict_i_3rd_neg, self.batch_loss_3rd_item_neg, self.loss_3rd_item_neg, self.optimizer_3rd_item_task_neg = self._3rd_item_task(
                name, self.support_user_3rd_neg, self.support_item_2nd_neg_, self.support_user_1st_neg_)

            self._2nd_pairwise_loss()
            self._3rd_pairwise_loss()

            # adaptive neighbor reviser
            self.global_step = tf.Variable(0, trainable=False, name="AgentStep")
            self.sess = sess
            self.lr = setting.learning_rate
            self.learning_rate_agent = tf.train.exponential_decay(self.lr, self.global_step, 10000, 0.95,
                                                                  staircase=True)
            self.tau = setting.agent_pretrain_tau
            self.second_order_state_size = setting.second_order_state_size
            self.third_order_state_size = setting.third_order_state_size
            self.weight_size = setting.agent_weight_size
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate_agent)
            self.num_other_variables = len(tf.trainable_variables())

            '''
            2nd neighbor network
            '''
            # Agent network(updating)
            self.second_order_input_state, _ = self.create_agent_network("Activate/second",
                                                                         self.second_order_state_size)
            self.second_order_network_params = tf.trainable_variables()[self.num_other_variables:]

            # Agent network(delayed updating)
            self.target_second_order_input_state, self.target_second_order_prob = self.create_agent_network(
                "Target/second",
                self.second_order_state_size)
            self.target_second_order_network_params = tf.trainable_variables()[
                                                      self.num_other_variables + len(self.second_order_network_params):]

            # delayed updaing Agent network
            self.update_target_second_order_network_params = \
                [self.target_second_order_network_params[i].assign(
                    tf.multiply(self.second_order_network_params[i], self.tau) + \
                    tf.multiply(self.target_second_order_network_params[i], 1 - self.tau)) \
                    for i in range(len(self.target_second_order_network_params))]

            self.assign_active_second_order_network_params = \
                [self.second_order_network_params[i].assign(
                    self.target_second_order_network_params[i]) for i in range(len(self.second_order_network_params))]

            self.second_order_reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
            self.second_order_action_holder = tf.placeholder(shape=[None], dtype=tf.float32)  #
            self.second_order_pi = self.second_order_action_holder * self.target_second_order_prob + (
                                                                                                         1 - self.second_order_action_holder) * (
                                                                                                     1 - self.target_second_order_prob)
            self.second_order_loss = -tf.reduce_sum(tf.log(self.second_order_pi) * self.second_order_reward_holder)
            self.second_order_gradients = tf.gradients(self.second_order_loss, self.target_second_order_network_params)

            self.second_order_grads = [tf.placeholder(tf.float32, [self.second_order_state_size, 1]),
                                       tf.placeholder(tf.float32, [1, 1])]

            # update parameters using gradient
            self.second_order_gradient_holders = []
            for idx, var in enumerate(self.second_order_network_params):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
                self.second_order_gradient_holders.append(placeholder)
            self.second_order_optimize = self.optimizer.apply_gradients(
                zip(self.second_order_gradient_holders, self.second_order_network_params),
                global_step=self.global_step)

            '''
            3rd neighbor network
            '''
            # Agent network(updating)
            self.third_order_input_state, _ = self.create_agent_network("Activate/third", self.third_order_state_size)
            self.third_order_network_params = tf.trainable_variables()[
                                              self.num_other_variables + len(self.second_order_network_params) + len(
                                                  self.target_second_order_network_params):]

            # Agent network(delayed updating)
            self.target_third_order_input_state, self.target_third_order_prob = self.create_agent_network(
                "Target/third", self.third_order_state_size)
            self.target_third_order_network_params = tf.trainable_variables()[self.num_other_variables + len(
                self.second_order_network_params) + len(self.target_second_order_network_params) + len(
                self.third_order_network_params):]

            # delayed updating Agent network
            self.update_target_third_order_network_params = \
                [self.target_third_order_network_params[i].assign(
                    tf.multiply(self.third_order_network_params[i], self.tau) + \
                    tf.multiply(self.target_third_order_network_params[i], 1 - self.tau)) \
                    for i in range(len(self.target_third_order_network_params))]

            self.assign_active_third_order_network_params = \
                [self.third_order_network_params[i].assign(
                    self.target_third_order_network_params[i]) for i in range(len(self.third_order_network_params))]

            self.third_order_reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
            self.third_order_action_holder = tf.placeholder(shape=[None], dtype=tf.float32)  #
            self.third_order_pi = self.third_order_action_holder * self.target_third_order_prob + (
                                                                                                      1 - self.third_order_action_holder) * (
                                                                                                  1 - self.target_third_order_prob)
            self.third_order_loss = -tf.reduce_sum(tf.log(self.third_order_pi) * self.third_order_reward_holder)
            self.third_order_gradients = tf.gradients(self.third_order_loss, self.target_third_order_network_params)

            self.third_order_grads = [tf.placeholder(tf.float32, [self.third_order_state_size, 1]),
                                      tf.placeholder(tf.float32, [1, 1])]

            # update parameters using gradient
            self.third_order_gradient_holders = []
            for idx, var in enumerate(self.third_order_network_params):
                placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
                self.third_order_gradient_holders.append(placeholder)
            self.third_order_optimize = self.optimizer.apply_gradients(
                zip(self.third_order_gradient_holders, self.third_order_network_params),
                global_step=self.global_step)

            self.original_user_ebd = np.load(setting.pre_train_user_ebd_path)
            self.original_item_ebd = np.load(setting.pre_train_item_ebd_path)
            embedding_size = self.original_user_ebd.shape[1]
            # padding
            padding_ebd = np.zeros((1, embedding_size), dtype=np.float32)
            self.original_user_ebd = np.concatenate((self.original_user_ebd, padding_ebd), 0)
            self.original_item_ebd = np.concatenate((self.original_item_ebd, padding_ebd), 0)

    def _create_placeholder(self):
        '''
        2nd task item
        '''
        # predict target item
        self.support_user = tf.placeholder(tf.int32, shape=[None, None])
        self.target_item = tf.placeholder(tf.float32, shape=[None, self.embedding_size])
        self.training_phrase_item_task = tf.placeholder(tf.bool, name='training-flag')
        self.test_support_user = tf.placeholder(tf.int32, shape=[1, None])

        # aggregation placeholder pos
        self.support_item_2nd_pos = tf.placeholder(tf.int32, shape=[None, None])
        self.support_user_1st_pos = tf.placeholder(tf.int32, shape=[None, None])
        # aggregation placeholder neg
        self.support_item_2nd_neg = tf.placeholder(tf.int32, shape=[None, None])
        self.support_user_1st_neg = tf.placeholder(tf.int32, shape=[None, None])

        '''
        2nd task user
        '''
        ########## predict target user task ########
        self.support_item = tf.placeholder(tf.int32, shape=[None, None])
        self.target_user = tf.placeholder(tf.float32, shape=[None, self.embedding_size])
        self.training_phrase_user_task = tf.placeholder(tf.bool, name='training-flag')
        self.test_support_item = tf.placeholder(tf.int32, shape=[1, None])

        # predict target user
        self.support_item_1st = tf.placeholder(tf.int32, shape=[None, None])
        self.support_user_2nd = tf.placeholder(tf.int32, shape=[None, None])

        '''
        3rd task user
        '''
        # predict target user
        self.support_item_3rd = tf.placeholder(tf.int32, shape=[None, None])
        self.support_user_2nd_ = tf.placeholder(tf.int32, shape=[None, None])
        self.support_item_1st_ = tf.placeholder(tf.int32, shape=[None, None])

        '''
        3rd task item
        '''
        # predict target item
        self.support_user_3rd_pos = tf.placeholder(tf.int32, shape=[None, None])
        self.support_item_2nd_pos_ = tf.placeholder(tf.int32, shape=[None, None])
        self.support_user_1st_pos_ = tf.placeholder(tf.int32, shape=[None, None])

        self.support_user_3rd_neg = tf.placeholder(tf.int32, shape=[None, None])
        self.support_item_2nd_neg_ = tf.placeholder(tf.int32, shape=[None, None])
        self.support_user_1st_neg_ = tf.placeholder(tf.int32, shape=[None, None])

        # fbne
        self.inter_support_2nd_item = tf.placeholder(tf.int32, shape=[None, None])
        self.inter_support_3rd_item = tf.placeholder(tf.int32, shape=[None, None])


    def udpate_tau(self, tau):
        self.tau = tau

    def update_lr(self, lr):
        self.lr = lr

    def create_agent_network(self, scope, state_size):
        with tf.name_scope(scope):
            input_state = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
            embedding_size = state_size
            weight_size = self.weight_size

            W = tf.Variable(tf.truncated_normal(shape=[embedding_size, weight_size], mean=0.0,
                                                stddev=tf.sqrt(tf.div(2.0, weight_size + embedding_size))),
                            name='Weights_for_MLP', dtype=tf.float32, trainable=True)
            
            b = tf.Variable(tf.constant(0, shape=[1, weight_size], dtype=tf.float32), name='Bias_for_MLP',
                            dtype=tf.float32, trainable=True)

            h = tf.Variable(
                tf.truncated_normal(shape=[weight_size, 1], mean=0.0, stddev=tf.sqrt(tf.div(2.0, weight_size))),
                name='H_for_MLP', dtype=tf.float32, trainable=True)

            MLP_output = tf.matmul(input_state, W) + b  # (b, e) * (e, w) + (1, w)
            MLP_output = tf.nn.relu(MLP_output)
            prob = tf.nn.sigmoid(tf.reduce_sum(tf.matmul(MLP_output, h), 1))  # (b, w) * (w,1 ) => (b, 1)
            prob = tf.clip_by_value(prob, 1e-5, 1 - 1e-5)
            return input_state, prob

    # second
    def init_second_order_gradbuffer(self):
        gradBuffer = self.sess.run(self.target_second_order_network_params)
        for index, grad in enumerate(gradBuffer):
            gradBuffer[index] = grad * 0
        return gradBuffer

    def train_second_order(self, second_order_gradbuffer, second_order_grads):
        for index, grad in enumerate(second_order_grads):
            second_order_gradbuffer[index] += grad
        feed_dict = dict(zip(self.second_order_gradient_holders, second_order_gradbuffer))
        self.sess.run(self.second_order_optimize, feed_dict=feed_dict)

    def predict_second_order_target(self, second_order_state):
        return self.sess.run(self.target_second_order_prob, feed_dict={
            self.target_second_order_input_state: second_order_state})

    def get_second_order_gradient(self, second_order_state, second_order_reward, second_order_action):
        return self.sess.run(self.second_order_gradients, feed_dict={
            self.target_second_order_input_state: second_order_state,
            self.second_order_reward_holder: second_order_reward,
            self.second_order_action_holder: second_order_action})

    def update_target_second_order_network(self):
        self.sess.run(self.update_target_second_order_network_params)

    def assign_active_second_order_network(self):
        self.sess.run(self.assign_active_second_order_network_params)

    # third
    def init_third_order_gradbuffer(self):
        gradBuffer = self.sess.run(self.target_third_order_network_params)
        for index, grad in enumerate(gradBuffer):
            gradBuffer[index] = grad * 0
        return gradBuffer

    def train_third_order(self, third_order_gradbuffer, third_order_grads):
        for index, grad in enumerate(third_order_grads):
            third_order_gradbuffer[index] += grad
        feed_dict = dict(zip(self.third_order_gradient_holders, third_order_gradbuffer))
        self.sess.run(self.third_order_optimize, feed_dict=feed_dict)

    def predict_third_order_target(self, third_order_state):
        return self.sess.run(self.target_third_order_prob, feed_dict={
            self.target_third_order_input_state: third_order_state})

    def get_third_order_gradient(self, third_order_state, third_order_reward, third_order_action):
        return self.sess.run(self.third_order_gradients, feed_dict={
            self.target_third_order_input_state: third_order_state,
            self.third_order_reward_holder: third_order_reward,
            self.third_order_action_holder: third_order_action})

    def update_target_third_order_network(self):
        self.sess.run(self.update_target_third_order_network_params)

    def assign_active_third_order_network(self):
        self.sess.run(self.assign_active_third_order_network_params)

    def _2nd_pairwise_loss(self):
        # variable
        self.second_w_pos = self.glorot(shape=[self.embedding_size, 50], name='second_w_pos')
        self.second_w_1 = self.glorot(shape=[50, self.embedding_size], name='second_w_neg')

        self.second_mf_loss, self.second_emb_loss, self.second_reg_loss = self._create_bpr_loss(
            self.predict_u_2nd,
            self.predict_i_2nd_pos,
            self.predict_i_2nd_neg, self.second_w_pos, self.second_w_1)

        self.second_loss = self.second_mf_loss + self.second_emb_loss
        self.second_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_downstream).minimize(self.second_loss)

    def _3rd_pairwise_loss(self):

        self.third_w_pos = self.glorot(shape=[self.embedding_size, 50], name='third_w_pos')

        self.third_w_pos1 = self.glorot(shape=[50, 1], name='third_w_pos_1')

        self.third_w_neg = self.glorot(shape=[self.embedding_size, self.embedding_size], name='third_w_neg')

        self.third_mf_loss, self.third_emb_loss, self.third_reg_loss = self._create_bpr_loss(
            self.predict_u_3rd,
            self.predict_i_3rd_pos,
            self.predict_i_3rd_neg, self.third_w_pos, self.third_w_pos1)

        self.third_loss = self.third_mf_loss + self.third_emb_loss
        self.third_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate_downstream).minimize(self.third_loss)

    def _create_bpr_loss(self, users, pos_items, neg_items, w1, w2):

        pos_scores = tf.reduce_sum(tf.multiply(users, pos_items), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(users, neg_items), axis=1)
        
        self.batch_predicts = pos_scores

        regularizer = tf.nn.l2_loss(users) + tf.nn.l2_loss(pos_items) + tf.nn.l2_loss(neg_items)
        regularizer = regularizer / self.batch_size
        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        emb_loss = self.decay * regularizer

        reg_loss = tf.constant(0.0, tf.float32, [1])

        return mf_loss, emb_loss, reg_loss

    def glorot(self, shape, name=None):
        """Glorot & Bengio (AISTATS 2010) init."""
        init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    def _1st_user_task(self):
        # user task
        self.pre_train_item_ebd_path = setting.pre_train_item_ebd_path

        final_support_encode_user_task = self._create_aggregator_network_user_task('active')

        batch_loss_user_task = Cosine_similarity(final_support_encode_user_task, self.target_user)

        loss_user_task = -tf.reduce_mean(Cosine_similarity(final_support_encode_user_task, self.target_user))

        optimizer_user_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                        initial_accumulator_value=1e-8).minimize(
            loss_user_task)

        return final_support_encode_user_task, batch_loss_user_task, loss_user_task, optimizer_user_task

    def _1st_item_task(self):
        # item_task
        self.pre_train_user_ebd_path = setting.pre_train_user_ebd_path

        final_support_encode_item_task = self._create_aggregator_network_item_task('active')

        batch_loss_item_task = Cosine_similarity(final_support_encode_item_task, self.target_item)

        loss_item_task = -tf.reduce_mean(Cosine_similarity(final_support_encode_item_task, self.target_item))

        optimizer_item_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                        initial_accumulator_value=1e-8).minimize(
            loss_item_task)

        return final_support_encode_item_task, batch_loss_item_task, loss_item_task, optimizer_item_task

    def _create_aggregator_network_item_task(self, scope):

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # with tf.name_scope(scope):
            pre_train_user_ebd = np.load(self.pre_train_user_ebd_path)
            self.c1 = tf.Variable(tf.constant(pre_train_user_ebd, shape=[self.num_users, self.embedding_size]),
                                  trainable=True)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.context_embedding_u = tf.concat([self.c1, self.c2], 0, name='embedding_user')

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # with tf.name_scope(scope):
            self.support_ebd = tf.nn.embedding_lookup(self.context_embedding_u, self.support_user)  # [b, n, e]
            self.support_encode_item = self._encode(self.support_ebd,
                                                    training=self.training_phrase_item_task)  # [b, n, e]
            final_support_encode_item_task = tf.reduce_mean(self.support_encode_item,
                                                            axis=1)  # [b, n, e] -> [b, e]
            return final_support_encode_item_task

    def _create_aggregator_network_user_task(self, scope):
        with tf.name_scope(scope):
            self.support_item = tf.placeholder(tf.int32, shape=[None, None])
            self.target_user = tf.placeholder(tf.float32, shape=[None, self.embedding_size])
            self.training_phrase_user_task = tf.placeholder(tf.bool, name='training-flag')
            self.test_support_item = tf.placeholder(tf.int32, shape=[1, None])

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            pre_train_item_ebd = np.load(self.pre_train_item_ebd_path)
            self.c1 = tf.Variable(tf.constant(pre_train_item_ebd, shape=[self.num_items, self.embedding_size]),
                                  trainable=True)
            self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
            self.context_embedding_i = tf.concat([self.c1, self.c2], 0, name='embedding_item')

        with tf.name_scope(scope):
            self.support_ebd = tf.nn.embedding_lookup(self.context_embedding_i, self.support_item)  # [b, n, e]
            self.support_encode_user = self._encode(self.support_ebd,
                                                    training=self.training_phrase_user_task)  # [b, n, e]
            final_support_encode_user_task = tf.reduce_mean(self.support_encode_user,
                                                            axis=1)  # [b, n, e] -> [b, e]

            return final_support_encode_user_task

    def _encode(self, input, training=True):
        '''
        input: [b, n, e]
        output: [b, n, e]
        '''
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            enc = input
            enc *= self.embedding_size ** 0.5

            ## Blocks
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.dropout_rate,
                                              training=training,
                                              causality=False)  # [b, q, e]
                    # feed forward
                    enc = ff(enc, num_units=[self.d_ff, self.embedding_size])  # [b, q, e]
        memory = enc
        return memory

    def _2nd_user_task(self, name, support_user_2nd, support_item_1st):
        if name == 'GAT':
            print('use GAT')
            with tf.name_scope('GAT_2nd_u'):
                with tf.variable_scope('GAT_2nd_u', reuse=tf.AUTO_REUSE):
                    w_0u = self.glorot([self.embedding_size, self.embedding_size], name='self_weights_0u')
                    w_1u = self.glorot([3 * self.embedding_size, self.embedding_size], name='self_weights_1u')

                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_u,
                                                                 support_user_2nd)  # [b, n2, e]
                    support_encode_2nd = tf.reduce_mean(
                        self._encode(support_ori_ebd_2nd, training=self.training_phrase_user_task), 1)  # [b, e]
                    ori_2nd_ebd = tf.reduce_mean(support_ori_ebd_2nd, 1)  # [b, e]

                   
                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_i,
                                                                 support_item_1st)  # [b, n1, e]
                    
                    ori_1st_ebd = tf.reduce_mean(support_ori_ebd_1st, 1)  # [b, e]
                    aggregate_2nd = tf.concat([support_encode_2nd, support_encode_2nd, ori_1st_ebd], 1)  # [b, 3e]

                    # perform aggregation
                    refined_first_neigh_ebd = tf.matmul(aggregate_2nd, w_1u)  # [b, 3e] * [3e, e] = [b, e]
                    refined_target_ebd = tf.matmul(refined_first_neigh_ebd, w_0u)  # [b, e]
                    

                    intra_info = refined_target_ebd
                    predict_u_2nd = intra_info

                    batch_loss_2nd_user = Cosine_similarity(predict_u_2nd, self.target_user)
                    loss_2nd_user = -tf.reduce_mean(Cosine_similarity(predict_u_2nd, self.target_user))
                    optimizer_2nd_user_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_2nd_user)
            return predict_u_2nd, batch_loss_2nd_user, loss_2nd_user, optimizer_2nd_user_task

        elif name == 'GraphSAGE':
            print('use GraphSAGE')
            with tf.name_scope('GraphSAGE_2nd_u'):
                with tf.variable_scope('GraphSAGE_2nd_u', reuse=tf.AUTO_REUSE):
                    w_0u = self.glorot([self.embedding_size, self.embedding_size], name='self_weights_0u')
                    w_1u = self.glorot([3 * self.embedding_size, self.embedding_size], name='self_weights_1u')

                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_u,
                                                                 support_user_2nd)  # [b, n2, e]
                    support_encode_2nd = tf.reduce_mean(
                        self._encode(support_ori_ebd_2nd, training=self.training_phrase_user_task), 1)  # [b, e]
                    ori_2nd_ebd = tf.reduce_mean(support_ori_ebd_2nd, 1)  # [b, e]

                    
                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_i,
                                                                 support_item_1st)  # [b, n1, e]
                   
                    ori_ebd_1st = tf.reduce_mean(support_ori_ebd_1st, 1)  # [b, e]


                    aggregate_2nd = tf.concat([support_encode_2nd, ori_2nd_ebd, ori_ebd_1st],
                                              1)  # [b, 3e]  second-order-neighbor || meta || first-order-neighbor


                    # perform aggregation
                    refined_first_neigh_ebd = tf.matmul(aggregate_2nd, w_1u)  # [b, 3e] * [3e, e] = [b, e]
                    refined_target_ebd = tf.matmul(refined_first_neigh_ebd, w_0u)  # [b, e]

                    intra_info = refined_target_ebd
                    predict_u_2nd = intra_info

                    batch_loss_2nd_user = Cosine_similarity(predict_u_2nd, self.target_user)
                    loss_2nd_user = -tf.reduce_mean(Cosine_similarity(predict_u_2nd, self.target_user))
                    optimizer_2nd_user_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_2nd_user)

            return predict_u_2nd, batch_loss_2nd_user, loss_2nd_user, optimizer_2nd_user_task



        elif name == 'FBNE':
            print('use FBNE')
            with tf.name_scope('FBNE_2nd_u'):
                with tf.variable_scope('FBNE_2nd_u', reuse=tf.AUTO_REUSE):
                    w_0u = self.glorot([self.embedding_size, self.embedding_size], name='self_weights_0u')
                    w_1u = self.glorot([3 * self.embedding_size, self.embedding_size], name='self_weights_1u')

                    w_concat = self.glorot([2 * self.embedding_size, self.embedding_size],
                                           name='concat_inter_intra_2nd_u')
                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_u,
                                                                 support_user_2nd)  # [b, n2, e]
                    support_encode_2nd = tf.reduce_mean(
                        self._encode(support_ori_ebd_2nd, training=self.training_phrase_user_task), 1)  # [b, e]
                    ori_2nd_ebd = tf.reduce_mean(support_ori_ebd_2nd, 1)  # [b, e]

                   
                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_i,
                                                                 support_item_1st)  # [b, n1, e]
                   
                    ori_ebd_1st = tf.reduce_mean(support_ori_ebd_1st, 1)  # [b, e]

                    aggregate_2nd = tf.concat([support_encode_2nd, ori_2nd_ebd, ori_ebd_1st],
                                              1)  # [b, 3e]  second-order-neighbor || meta || first-order-neighbor


                    # perform aggregation
                    refined_first_neigh_ebd = tf.matmul(aggregate_2nd, w_1u)  # [b, 3e] * [3e, e] = [b, e]
                    refined_target_ebd = tf.matmul(refined_first_neigh_ebd, w_0u)  # [b, e]

                    intra_info = refined_target_ebd  # [b, e]

                    self.inter_support_2nd_user = tf.placeholder(tf.int32, shape=[None,
                                                                                  None])  # [b, n3] corresponding to the target user
                    support_inter_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_u,
                                                                       self.inter_support_2nd_user)  # [b, n3, e]
                    support_encode_inter_ebd_2nd = tf.reduce_mean(self._encode(support_inter_ori_ebd_2nd), 1)  # [b, e]
                    inter_info = support_encode_inter_ebd_2nd

                    # equation 9 FNBE
                    predict_u_2nd = tf.matmul(tf.concat([intra_info, inter_info], 1),
                                              w_concat)  # [b, 2e]


                    batch_loss_2nd_user = Cosine_similarity(predict_u_2nd, self.target_user)
                    loss_2nd_user = -tf.reduce_mean(Cosine_similarity(predict_u_2nd, self.target_user))
                    optimizer_2nd_user_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_2nd_user)

            return predict_u_2nd, batch_loss_2nd_user, loss_2nd_user, optimizer_2nd_user_task

        elif name == 'LightGCN':
            print('use LightGCN')
            with tf.name_scope('LightGCN_2nd_u'):
                with tf.variable_scope('LightGCN_2nd_u', reuse=tf.AUTO_REUSE):
                    w_0u = self.glorot([2 * self.embedding_size, self.embedding_size], name='self_weights_0u')
                    # pre-process matrix
                    # AX item
                    pre_train_item_ebd = np.load(self.pre_train_i_ax)
                    c1 = tf.Variable(tf.constant(pre_train_item_ebd, shape=[self.num_items, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2ax')
                    self.context_embedding_i_normalized_1st = tf.concat([c1, c2], 0, name='embedding_item_ax')

                    # AX user
                    pre_train_user_ebd = np.load(self.pre_train_u_ax)
                    c1 = tf.Variable(tf.constant(pre_train_user_ebd, shape=[self.num_users, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
                    self.context_embedding_u_normalized_1st = tf.concat([c1, c2], 0, name='embedding_user_ax')

                    # AAX item
                    pre_train_item_ebd = np.load(self.pre_train_i_aax)
                    c1 = tf.Variable(tf.constant(pre_train_item_ebd, shape=[self.num_items, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2ax')
                    self.context_embedding_i_normalized_2nd = tf.concat([c1, c2], 0, name='embedding_item_ax')

                    # AAX user
                    pre_train_user_ebd = np.load(self.pre_train_u_aax)
                    c1 = tf.Variable(tf.constant(pre_train_user_ebd, shape=[self.num_users, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
                    self.context_embedding_u_normalized_2nd = tf.concat([c1, c2], 0, name='embedding_user_ax')

                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_u_normalized_2nd,
                                                                 support_user_2nd)  # [b, n2, e]

                    ori_2nd_ebd = tf.reduce_mean(support_ori_ebd_2nd, 1)  # [b, e]

                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_i_normalized_1st,
                                                                 support_item_1st)  # [b, n1, e]
                    ori_1st_ebd = tf.reduce_sum(support_ori_ebd_1st, 1)  # [b, e]

                    aggregate_2nd = tf.concat(
                        [tf.reduce_sum(self._encode(support_ori_ebd_2nd, training=self.training_phrase_user_task), 1),
                         ori_2nd_ebd],
                        1)  # [b, 2e]  second-order-neighbor || meta
                    aggregate_1st = tf.concat(
                        [tf.reduce_sum(self._encode(support_ori_ebd_1st, training=self.training_phrase_item_task), 1),
                         ori_1st_ebd], 1)  # [b, 2e]

                    # perform aggregation
                    refined_target_ebd = tf.matmul(aggregate_2nd + aggregate_1st, w_0u)  # [b, 2e] * [2e, e] = [b, e]

                    intra_info = refined_target_ebd
                    predict_u_2nd = intra_info

                    batch_loss_2nd_user = Cosine_similarity(predict_u_2nd, self.target_user)
                    loss_2nd_user = -tf.reduce_mean(Cosine_similarity(predict_u_2nd, self.target_user))
                    optimizer_2nd_user_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_2nd_user)

            return predict_u_2nd, batch_loss_2nd_user, loss_2nd_user, optimizer_2nd_user_task


    def _2nd_item_task(self, name, support_item_2nd, support_user_1st):
        if name == 'GAT':
            print('use GAT')
            with tf.name_scope('GAT_2nd_i'):
                with tf.variable_scope('GAT_2nd_i', reuse=tf.AUTO_REUSE):
                    w_0i = self.glorot([self.embedding_size, self.embedding_size], name='self_weights_0i')
                    w_1i = self.glorot([3 * self.embedding_size, self.embedding_size], name='self_weights_1i')

                   
                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_i, support_item_2nd)
                    support_encode_2nd = tf.reduce_mean(
                        self._encode(support_ori_ebd_2nd, training=self.training_phrase_item_task), 1)  # [b ,e]
                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_u, support_user_1st)
                    support_encode_1st = tf.reduce_mean(
                        self._encode(support_ori_ebd_1st, training=self.training_phrase_user_task), 1)

                    ori_1st_ebd = tf.reduce_mean(support_ori_ebd_1st, 1)

                    # perform aggregation
                    refined_first_neigh_ebd = tf.matmul(
                        tf.concat([support_encode_2nd, support_encode_2nd, ori_1st_ebd], 1), w_1i)
                    refined_target_ebd = tf.matmul(refined_first_neigh_ebd, w_0i)

                    intra_info = refined_target_ebd
                    predict_i_2nd = intra_info

                    batch_loss_2nd_item = Cosine_similarity(predict_i_2nd, self.target_item)
                    loss_2nd_item = -tf.reduce_mean(Cosine_similarity(predict_i_2nd, self.target_item))
                    optimizer_2nd_item_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_2nd_item)

            return predict_i_2nd, batch_loss_2nd_item, loss_2nd_item, optimizer_2nd_item_task


        elif name == 'GraphSAGE':
            print('use GraphSAGE')
            with tf.name_scope('GraphSAGE_2nd_i'):
                with tf.variable_scope('GraphSAGE_2nd_i', reuse=tf.AUTO_REUSE):
                    w_0i = self.glorot([self.embedding_size, self.embedding_size], name='self_weights_0i')
                    w_1i = self.glorot([3 * self.embedding_size, self.embedding_size], name='self_weights_1i')

                  
                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_i, support_item_2nd)
                    support_encode_2nd = tf.reduce_mean(
                        self._encode(support_ori_ebd_2nd, training=self.training_phrase_item_task), 1)
                    ori_2nd_ebd = tf.reduce_mean(support_ori_ebd_2nd, 1)  # [b, e]

                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_u, support_user_1st)

                    ori_ebd_1st = tf.reduce_mean(support_ori_ebd_1st, 1)
                    aggregate_2nd = tf.concat([support_encode_2nd, ori_2nd_ebd, ori_ebd_1st],
                                              1)  # [b, 3e]  second-order-neighbor || meta || first-order-neighbor

                    # perform aggregation
                    refined_first_neigh_ebd = tf.matmul(aggregate_2nd, w_1i)  # [b, 3e] * [3e, e] = [b, e]
                    refined_target_ebd = tf.matmul(refined_first_neigh_ebd, w_0i)

                    intra_info = refined_target_ebd
                    predict_i_2nd = intra_info

                    batch_loss_2nd_item = Cosine_similarity(predict_i_2nd, self.target_item)
                    loss_2nd_item = -tf.reduce_mean(Cosine_similarity(predict_i_2nd, self.target_item))
                    optimizer_2nd_item_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_2nd_item)

            return predict_i_2nd, batch_loss_2nd_item, loss_2nd_item, optimizer_2nd_item_task


        elif name == 'FBNE':
            print('use FBNE')
            with tf.name_scope('FBNE_2nd_i'):
                with tf.variable_scope('FBNE_2nd_i', reuse=tf.AUTO_REUSE):
                    w_0i = self.glorot([self.embedding_size, self.embedding_size], name='self_weights_0i')
                    w_1i = self.glorot([3 * self.embedding_size, self.embedding_size], name='self_weights_1i')

                    w_concat = self.glorot([2 * self.embedding_size, self.embedding_size],
                                           name='concat_inter_intra_2nd_i')

                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_i, support_item_2nd)
                    support_encode_2nd = tf.reduce_mean(
                        self._encode(support_ori_ebd_2nd, training=self.training_phrase_item_task), 1)
                    ori_2nd_ebd = tf.reduce_mean(support_ori_ebd_2nd, 1)  # [b, e]

                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_u, support_user_1st)

                    ori_ebd_1st = tf.reduce_mean(support_ori_ebd_1st, 1)
                    aggregate_2nd = tf.concat([support_encode_2nd, ori_2nd_ebd, ori_ebd_1st],
                                              1)  # [b, 3e]  second-order-neighbor || meta || first-order-neighbor

                    # perform aggregation
                    refined_first_neigh_ebd = tf.matmul(aggregate_2nd, w_1i)  # [b, 3e] * [3e, e] = [b, e]
                    refined_target_ebd = tf.matmul(refined_first_neigh_ebd, w_0i)


                    intra_info = refined_target_ebd  # [b, e]
                    support_inter_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_i,
                                                                       self.inter_support_2nd_item)
                    support_encode_inter_ebd_2nd = tf.reduce_mean(
                        self._encode(support_inter_ori_ebd_2nd, training=self.training_phrase_item_task), 1)  # [b, e]
                    inter_info = support_encode_inter_ebd_2nd

                    # equation 9 FNBE
                    predict_i_2nd = tf.matmul(tf.concat([intra_info, inter_info], 1), w_concat)  # [b, e]

                    batch_loss_2nd_item = Cosine_similarity(predict_i_2nd, self.target_item)
                    loss_2nd_item = -tf.reduce_mean(Cosine_similarity(predict_i_2nd, self.target_item))
                    optimizer_2nd_item_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_2nd_item)

            return predict_i_2nd, batch_loss_2nd_item, loss_2nd_item, optimizer_2nd_item_task


        elif name == 'LightGCN':
            print('use LightGCN')
            with tf.name_scope('LightGCN_2nd_i'):
                with tf.variable_scope('LightGCN_2nd_i', reuse=tf.AUTO_REUSE):
                    w_0u = self.glorot([2 * self.embedding_size, self.embedding_size], name='self_weights_0u')
                    # pre-process matrix
                    # AX item
                    pre_train_item_ebd = np.load(self.pre_train_i_ax)
                    c1 = tf.Variable(tf.constant(pre_train_item_ebd, shape=[self.num_items, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2ax')
                    self.context_embedding_i_normalized_1st = tf.concat([c1, c2], 0, name='embedding_item_ax')

                    # AX user
                    pre_train_user_ebd = np.load(self.pre_train_u_ax)
                    c1 = tf.Variable(tf.constant(pre_train_user_ebd, shape=[self.num_users, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
                    self.context_embedding_u_normalized_1st = tf.concat([c1, c2], 0, name='embedding_user_ax')

                    # AAX item
                    pre_train_item_ebd = np.load(self.pre_train_i_aax)
                    c1 = tf.Variable(tf.constant(pre_train_item_ebd, shape=[self.num_items, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2ax')
                    self.context_embedding_i_normalized_2nd = tf.concat([c1, c2], 0, name='embedding_item_ax')

                    # AAX user
                    pre_train_user_ebd = np.load(self.pre_train_u_aax)
                    c1 = tf.Variable(tf.constant(pre_train_user_ebd, shape=[self.num_users, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
                    self.context_embedding_u_normalized_2nd = tf.concat([c1, c2], 0, name='embedding_user_ax')

                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_i_normalized_2nd,
                                                                 support_item_2nd)
                    ori_2nd_ebd = tf.reduce_mean(support_ori_ebd_2nd, 1)  # [b, e]

                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_u_normalized_1st,
                                                                 support_user_1st)
                    ori_1st_ebd = tf.reduce_sum(support_ori_ebd_1st, 1)  # [b, e]

                    aggregate_2nd = tf.concat(
                        [tf.reduce_sum(self._encode(support_ori_ebd_2nd, training=self.training_phrase_item_task), 1),
                         ori_2nd_ebd], 1)
                    aggregate_1st = tf.concat(
                        [tf.reduce_sum(self._encode(support_ori_ebd_1st, training=self.training_phrase_item_task), 1),
                         ori_1st_ebd], 1)

                    refined_target_ebd = tf.matmul(aggregate_2nd + aggregate_1st, w_0u)

                    intra_info = refined_target_ebd
                    predict_i_2nd = intra_info

                    batch_loss_2nd_item = Cosine_similarity(predict_i_2nd, self.target_item)
                    loss_2nd_item = -tf.reduce_mean(Cosine_similarity(predict_i_2nd, self.target_item))
                    optimizer_2nd_item_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_2nd_item)

            return predict_i_2nd, batch_loss_2nd_item, loss_2nd_item, optimizer_2nd_item_task


    def _3rd_item_task(self, name, support_user_3rd, support_item_2nd_, support_user_1st_):
        if name == 'GAT':
            print('use GAT')
            with tf.name_scope('GAT_3rd_i'):
                with tf.variable_scope('GAT_3rd_i', reuse=tf.AUTO_REUSE):
                    w_0i = self.glorot([self.embedding_size, self.embedding_size], name='weight_0i')
                    w_1i = self.glorot([3 * self.embedding_size, self.embedding_size], name='weight_1i')
                    w_2i = self.glorot([3 * self.embedding_size, self.embedding_size], name='weight_2i')

                    support_ori_ebd_3rd = tf.nn.embedding_lookup(self.context_embedding_u,
                                                                 support_user_3rd)  # [b, n, e]
                    support_encode_3rd = tf.reduce_mean(self._encode(support_ori_ebd_3rd,
                                                                     training=self.training_phrase_user_task),
                                                        1)  # b, e

                    
                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_i,
                                                                 support_item_2nd_)  # [b, n, e]
                    support_encode_2nd = tf.reduce_mean(
                        self._encode(support_ori_ebd_2nd, training=self.training_phrase_item_task), 1)

                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_u, support_user_1st_)
                    support_encode_1st = tf.reduce_mean(
                        self._encode(support_ori_ebd_1st, training=self.training_phrase_user_task), 1)

                    ori_2nd_ebd = tf.reduce_mean(support_ori_ebd_2nd, 1)  # b, e
                    ori_1st_ebd = tf.reduce_mean(support_ori_ebd_1st, 1)  # b, e

                    # perform aggregation
                    # 3order-2order
                    aggregate_3rd = tf.concat([support_encode_3rd, support_encode_3rd,
                                               ori_2nd_ebd],
                                              1)  # [b, 3e] third-order-neighbor || meta || second-order-neighbor
                    refined_second_neigh_ebd = tf.matmul(aggregate_3rd, w_2i)  # [b, 3e] * [3e, e] = [b, e]
                    # 2order-1order
                    aggregate_2nd = tf.concat([refined_second_neigh_ebd, support_encode_2nd,
                                               ori_1st_ebd],
                                              1)  # [b, 3e] second-order-neighbor || meta || first-order-neighbor
                    refined_first_neigh_ebd = tf.matmul(aggregate_2nd, w_1i)  # [b, 3e] * [3d, e] = [b, e]
                    refined_target_ebd = tf.matmul(refined_first_neigh_ebd, w_0i)  # [b, e] * [e, e] = [b, e]

                    intra_info = refined_target_ebd  # b, e
                    predict_i_3rd = intra_info


                    batch_loss_3rd_item = Cosine_similarity(predict_i_3rd, self.target_item)
                    loss_3rd_item = -tf.reduce_mean(Cosine_similarity(predict_i_3rd, self.target_item))
                    optimizer_3rd_item_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_3rd_item)

            return predict_i_3rd, batch_loss_3rd_item, loss_3rd_item, optimizer_3rd_item_task

        elif name == 'GraphSAGE':
            print('use GraphSAGE')
            with tf.name_scope('GraphSAGE_3rd_i'):
                with tf.variable_scope('GraphSAGE_3rd_i', reuse=tf.AUTO_REUSE):
                   
                    w_0i = self.glorot([self.embedding_size, self.embedding_size], name='weight_0i')
                    w_1i = self.glorot([3 * self.embedding_size, self.embedding_size], name='weight_1i')
                    w_2i = self.glorot([3 * self.embedding_size, self.embedding_size], name='weight_2i')
                    support_ori_ebd_3rd = tf.nn.embedding_lookup(self.context_embedding_u,
                                                                 support_user_3rd)  # [b, n, e]
                    support_encode_3rd = tf.reduce_mean(self._encode(support_ori_ebd_3rd,
                                                                     training=self.training_phrase_user_task),
                                                        1)  # b, e

                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_i,
                                                                 support_item_2nd_)  # [b, n, e]
                    support_encode_2nd = tf.reduce_mean(
                        self._encode(support_ori_ebd_2nd, training=self.training_phrase_item_task), 1)

                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_u, support_user_1st_)
                    support_encode_1st = tf.reduce_mean(
                        self._encode(support_ori_ebd_1st, training=self.training_phrase_user_task), 1)

                    ori_3rd_ebd = tf.reduce_mean(support_ori_ebd_3rd, 1)  # [b, e]
                    ori_2nd_ebd = tf.reduce_mean(support_ori_ebd_2nd, 1)  # [b, e]
                    ori_1st_ebd = tf.reduce_mean(support_ori_ebd_1st, 1)  # [b, e]

                    # perform aggregation
                    # 3order-2order
                    aggregate_3rd = tf.concat([support_encode_3rd, ori_3rd_ebd, ori_2nd_ebd],
                                              1)  # [b, 3e] third-order-neighbor || meta || second-order-neighbor
                    refined_second_neigh_ebd = tf.matmul(aggregate_3rd, w_2i)  # [b, 3e] * [3e, e] = [b, e]
                    # 2order-1order
                    aggregate_2nd = tf.concat([refined_second_neigh_ebd, support_encode_2nd,
                                               ori_1st_ebd],
                                              1)  # [b, 3e] second-order-neighbor || meta || first-order-neighbor
                    refined_first_neigh_ebd = tf.matmul(aggregate_2nd, w_1i)  # [b, 3e] * [3d, e] = [b, e]
                    refined_target_ebd = tf.matmul(refined_first_neigh_ebd, w_0i)  # [b, e] * [e, e] = [b, e]


                    intra_info = refined_target_ebd  # b, e
                    predict_i_3rd = intra_info

                    batch_loss_3rd_item = Cosine_similarity(predict_i_3rd, self.target_item)
                    loss_3rd_item = -tf.reduce_mean(Cosine_similarity(predict_i_3rd, self.target_item))
                    optimizer_3rd_item_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_3rd_item)

            return predict_i_3rd, batch_loss_3rd_item, loss_3rd_item, optimizer_3rd_item_task


        elif name == 'FBNE':
            print('use FBNE')
            with tf.name_scope('FBNE_3rd_i'):
                with tf.variable_scope('FBNE_3rd_i', reuse=tf.AUTO_REUSE):
                    w_0i = self.glorot([self.embedding_size, self.embedding_size], name='weight_0i')
                    w_1i = self.glorot([3 * self.embedding_size, self.embedding_size], name='weight_1i')
                    w_2i = self.glorot([3 * self.embedding_size, self.embedding_size], name='weight_2i')

                    w_concat = self.glorot([2 * self.embedding_size, self.embedding_size],
                                           name='concat_inter_intra_3rd_i')

                   
                    support_ori_ebd_3rd = tf.nn.embedding_lookup(self.context_embedding_u,
                                                                 support_user_3rd)  # [b, n, e]
                    support_encode_3rd = tf.reduce_mean(self._encode(support_ori_ebd_3rd,
                                                                     training=self.training_phrase_user_task),
                                                        1)  # b, e

                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_i,
                                                                 support_item_2nd_)  # [b, n, e]
                    support_encode_2nd = tf.reduce_mean(
                        self._encode(support_ori_ebd_2nd, training=self.training_phrase_item_task), 1)

                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_u, support_user_1st_)
                    support_encode_1st = tf.reduce_mean(
                        self._encode(support_ori_ebd_1st, training=self.training_phrase_user_task), 1)

                    ori_3rd_ebd = tf.reduce_mean(support_ori_ebd_3rd, 1)  # [b, e]
                    ori_2nd_ebd = tf.reduce_mean(support_ori_ebd_2nd, 1)  # [b, e]
                    ori_1st_ebd = tf.reduce_mean(support_ori_ebd_1st, 1)  # [b, e]

                    # perform aggregation
                    # 3order-2order
                    aggregate_3rd = tf.concat([support_encode_3rd, ori_3rd_ebd, ori_2nd_ebd],
                                              1)  # [b, 3e] third-order-neighbor || meta || second-order-neighbor
                    refined_second_neigh_ebd = tf.matmul(aggregate_3rd, w_2i)  # [b, 3e] * [3e, e] = [b, e]
                    # 2order-1order
                    aggregate_2nd = tf.concat([refined_second_neigh_ebd, support_encode_2nd,
                                               ori_1st_ebd],
                                              1)  # [b, 3e] second-order-neighbor || meta || first-order-neighbor
                    refined_first_neigh_ebd = tf.matmul(aggregate_2nd, w_1i)  # [b, 3e] * [3d, e] = [b, e]
                    refined_target_ebd = tf.matmul(refined_first_neigh_ebd, w_0i)  # [b, e] * [e, e] = [b, e]

                    intra_info = refined_target_ebd  # b, e

                    support_inter_ori_ebd_3rd = tf.nn.embedding_lookup(self.context_embedding_i,
                                                                       self.inter_support_3rd_item)
                    support_encode_inter_ebd_3rd = tf.reduce_mean(
                        self._encode(support_inter_ori_ebd_3rd, training=self.training_phrase_item_task), 1)
                    inter_info = support_encode_inter_ebd_3rd

                    # equation 9 FBNE
                    predict_i_3rd = tf.matmul(tf.concat([intra_info, inter_info], 1),
                                              w_concat)  # [b, 2e] * [2e, e] = [b, e]

                    batch_loss_3rd_item = Cosine_similarity(predict_i_3rd, self.target_item)
                    loss_3rd_item = -tf.reduce_mean(Cosine_similarity(predict_i_3rd, self.target_item))
                    optimizer_3rd_item_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_3rd_item)

            return predict_i_3rd, batch_loss_3rd_item, loss_3rd_item, optimizer_3rd_item_task

        elif name == 'LightGCN':
            print('use LightGCN')
            with tf.name_scope('LightGCN_3rd_i'):
                with tf.variable_scope('LightGCN_3rd_i', reuse=tf.AUTO_REUSE):
                    w_0i = self.glorot([2 * self.embedding_size, self.embedding_size], name='self_weights_0u')
                    # pre-process matrix
                    # AX item
                    pre_train_item_ebd = np.load(self.pre_train_i_ax)
                    c1 = tf.Variable(tf.constant(pre_train_item_ebd, shape=[self.num_items, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2ax')
                    self.context_embedding_i_normalized_1st = tf.concat([c1, c2], 0, name='embedding_item_ax')

                    # AX user
                    pre_train_user_ebd = np.load(self.pre_train_u_ax)
                    c1 = tf.Variable(tf.constant(pre_train_user_ebd, shape=[self.num_users, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2ax')
                    self.context_embedding_u_normalized_1st = tf.concat([c1, c2], 0, name='embedding_user_ax')

                    # AAX item
                    pre_train_item_ebd = np.load(self.pre_train_i_aax)
                    c1 = tf.Variable(tf.constant(pre_train_item_ebd, shape=[self.num_items, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2aax')
                    self.context_embedding_i_normalized_2nd = tf.concat([c1, c2], 0, name='embedding_item_aax')

                    # AAX user
                    pre_train_user_ebd = np.load(self.pre_train_u_aax)
                    c1 = tf.Variable(tf.constant(pre_train_user_ebd, shape=[self.num_users, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2aax')
                    self.context_embedding_u_normalized_2nd = tf.concat([c1, c2], 0, name='embedding_user_aax')

                    # AAAX user
                    pre_train_user_ebd = np.load(self.pre_train_u_aaax)
                    c1 = tf.Variable(tf.constant(pre_train_user_ebd, shape=[self.num_users, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2aaax')
                    self.context_embedding_u_normalized_3rd = tf.concat([c1, c2], 0, name='embedding_user_aaax')

                    # AAAX item
                    pre_train_item_ebd = np.load(self.pre_train_i_aaax)
                    c1 = tf.Variable(tf.constant(pre_train_item_ebd, shape=[self.num_items, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2aaax')
                    self.context_embedding_i_normalized_3rd = tf.concat([c1, c2], 0, name='embedding_item_aaax')

                    # self.support_user_3rd = tf.placeholder(tf.int32, shape=[None, None])
                    support_ori_ebd_3rd = tf.nn.embedding_lookup(self.context_embedding_u_normalized_3rd,
                                                                 support_user_3rd)
                    ori_3rd_ebd = tf.reduce_mean(support_ori_ebd_3rd, 1)

                    # self.support_item_2nd_ = tf.placeholder(tf.int32, shape=[None, None])
                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_i_normalized_2nd,
                                                                 support_item_2nd_)
                    ori_2nd_ebd = tf.reduce_mean(support_ori_ebd_2nd, 1)

                    # self.support_user_1st_ = tf.placeholder(tf.int32, shape=[None, None])
                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_u_normalized_1st,
                                                                 support_user_1st_)
                    ori_1st_ebd = tf.reduce_mean(support_ori_ebd_1st, 1)

                    aggregate_3rd = tf.concat(
                        [tf.reduce_sum(self._encode(support_ori_ebd_3rd, training=self.training_phrase_item_task), 1),
                         ori_3rd_ebd], 1)
                    aggregate_2nd = tf.concat(
                        [tf.reduce_sum(self._encode(support_ori_ebd_2nd, training=self.training_phrase_item_task), 1),
                         ori_2nd_ebd], 1)
                    aggregate_1st = tf.concat(
                        [tf.reduce_sum(self._encode(support_ori_ebd_1st, training=self.training_phrase_item_task), 1),
                         ori_1st_ebd], 1)

                    refined_target_ebd = tf.matmul(aggregate_3rd + aggregate_2nd + aggregate_1st, w_0i)

                    intra_info = refined_target_ebd
                    predict_i_3rd = intra_info
                    batch_loss_3rd_item = Cosine_similarity(predict_i_3rd, self.target_item)
                    loss_3rd_item = -tf.reduce_mean(Cosine_similarity(predict_i_3rd, self.target_item))
                    optimizer_3rd_item_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_3rd_item)

            return predict_i_3rd, batch_loss_3rd_item, loss_3rd_item, optimizer_3rd_item_task


    def _3rd_user_task(self, name, support_item_3rd, support_user_2nd_, support_item_1st_):
        if name == 'GAT':
            print('use GAT')
            with tf.name_scope('GAT_3rd_u'):
                with tf.variable_scope('GAT_3rd_u', reuse=tf.AUTO_REUSE):
                    w_0u = self.glorot([self.embedding_size, self.embedding_size], name='weight_0u')
                    w_1u = self.glorot([3 * self.embedding_size, self.embedding_size], name='weight_1u')
                    w_2u = self.glorot([3 * self.embedding_size, self.embedding_size], name='weight_2u')
                    support_ori_ebd_3rd = tf.nn.embedding_lookup(self.context_embedding_i,
                                                                 support_item_3rd)  # [b, n, e]
                    support_encode_3rd = tf.reduce_mean(self._encode(support_ori_ebd_3rd,
                                                                     training=self.training_phrase_item_task),
                                                        1)  # b, e
                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_u,
                                                                 support_user_2nd_)  # [b, n, e]
                    support_encode_2nd = tf.reduce_mean(
                        self._encode(support_ori_ebd_2nd, training=self.training_phrase_user_task), 1)

                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_i, support_item_1st_)
                    support_encode_1st = tf.reduce_mean(
                        self._encode(support_ori_ebd_1st, training=self.training_phrase_item_task), 1)

                    ori_2nd_ebd = tf.reduce_mean(support_ori_ebd_2nd, 1)  # b, e
                    ori_1st_ebd = tf.reduce_mean(support_ori_ebd_1st, 1)  # b, e

                    # perform aggregation
                    # 3order-2order
                    aggregate_3rd = tf.concat([support_encode_3rd, support_encode_3rd, ori_2nd_ebd],
                                              1)  # [b, 3e] third-order-neighbor || meta || second-order-neighbor
                    refined_second_neigh_ebd = tf.matmul(aggregate_3rd, w_2u)  # [b, 3e] * [3, e] = [b, e]

                    # 2order-1order
                    aggregate_2nd = tf.concat([refined_second_neigh_ebd, support_encode_2nd, ori_1st_ebd],
                                              1)  # [b, 3e] second-order-neighbor || meta || first-order-neighbor
                    refined_first_neigh_ebd = tf.matmul(aggregate_2nd, w_1u)  # [b, 3e] * [3e, e] = [b, e]
                    refined_target_ebd = tf.matmul(refined_first_neigh_ebd, w_0u)

                    intra_info = refined_target_ebd  # b, e
                    predict_u_3rd = intra_info


                    batch_loss_3rd_user = Cosine_similarity(predict_u_3rd, self.target_user)
                    loss_3rd_user = -tf.reduce_mean(Cosine_similarity(predict_u_3rd, self.target_user))
                    optimizer_3rd_user_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_3rd_user)

            return predict_u_3rd, batch_loss_3rd_user, loss_3rd_user, optimizer_3rd_user_task


        elif name == 'GraphSAGE':
            print('use GraphSAGE')
            with tf.name_scope('GraphSAGE_3rd_u'):
                with tf.variable_scope('GraphSAGE_3rd_u', reuse=tf.AUTO_REUSE):
                    w_0u = self.glorot([self.embedding_size, self.embedding_size], name='weight_0u')
                    w_1u = self.glorot([3 * self.embedding_size, self.embedding_size], name='weight_1u')
                    w_2u = self.glorot([3 * self.embedding_size, self.embedding_size], name='weight_2u')
                    support_ori_ebd_3rd = tf.nn.embedding_lookup(self.context_embedding_i,
                                                                 support_item_3rd)  # [b, n, e]
                    support_encode_3rd = tf.reduce_mean(self._encode(support_ori_ebd_3rd,
                                                                     training=self.training_phrase_item_task),
                                                        1)  # b, e
                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_u,
                                                                 support_user_2nd_)  # [b, n, e]
                    support_encode_2nd = tf.reduce_mean(
                        self._encode(support_ori_ebd_2nd, training=self.training_phrase_user_task), 1)

                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_i, support_item_1st_)
                    support_encode_1st = tf.reduce_mean(
                        self._encode(support_ori_ebd_1st, training=self.training_phrase_item_task), 1)

                    ori_3rd_ebd = tf.reduce_mean(support_ori_ebd_3rd, 1)  # [b, e]
                    ori_2nd_ebd = tf.reduce_mean(support_ori_ebd_2nd, 1)  # [b, e]
                    ori_1st_ebd = tf.reduce_mean(support_ori_ebd_1st, 1)  # [b, e]

                    # perform aggregation
                    # 3order-2order
                    aggregate_3rd = tf.concat([support_encode_3rd, ori_3rd_ebd, ori_2nd_ebd],
                                              1)  # [b, 3e] third-order-neighbor || meta || second-order-neighbor
                    refined_second_neigh_ebd = tf.matmul(aggregate_3rd, w_2u)  # [b, 3e] * [3, e] = [b, e]

                    # 2order-1order
                    aggregate_2nd = tf.concat([refined_second_neigh_ebd, support_encode_2nd, ori_1st_ebd],
                                              1)  # [b, 3e] second-order-neighbor || meta || first-order-neighbor
                    refined_first_neigh_ebd = tf.matmul(aggregate_2nd, w_1u)  # [b, 3e] * [3e, e] = [b, e]
                    refined_target_ebd = tf.matmul(refined_first_neigh_ebd, w_0u)

                    intra_info = refined_target_ebd  # b, e
                    predict_u_3rd = intra_info


                    batch_loss_3rd_user = Cosine_similarity(predict_u_3rd, self.target_user)
                    loss_3rd_user = -tf.reduce_mean(Cosine_similarity(predict_u_3rd, self.target_user))
                    optimizer_3rd_user_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_3rd_user)

            return predict_u_3rd, batch_loss_3rd_user, loss_3rd_user, optimizer_3rd_user_task


        elif name == 'FBNE':
            print('use FBNE')
            with tf.name_scope('FNBE_3rd_u'):
                with tf.variable_scope('FBNE_3rd_u', reuse=tf.AUTO_REUSE):
                    w_0u = self.glorot([self.embedding_size, self.embedding_size], name='weight_0u')
                    w_1u = self.glorot([3 * self.embedding_size, self.embedding_size], name='weight_1u')
                    w_2u = self.glorot([3 * self.embedding_size, self.embedding_size], name='weight_2u')

                    w_concate = self.glorot([2 * self.embedding_size, self.embedding_size],
                                            name='concat_inter_intra_3rd_u')

                    support_ori_ebd_3rd = tf.nn.embedding_lookup(self.context_embedding_i,
                                                                 support_item_3rd)  # [b, n, e]
                    support_encode_3rd = tf.reduce_mean(self._encode(support_ori_ebd_3rd,
                                                                     training=self.training_phrase_item_task),
                                                        1)  # b, e
                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_u,
                                                                 support_user_2nd_)  # [b, n, e]
                    support_encode_2nd = tf.reduce_mean(
                        self._encode(support_ori_ebd_2nd, training=self.training_phrase_user_task), 1)

                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_i, support_item_1st_)
                    support_encode_1st = tf.reduce_mean(
                        self._encode(support_ori_ebd_1st, training=self.training_phrase_user_task), 1)

                    ori_3rd_ebd = tf.reduce_mean(support_ori_ebd_3rd, 1)  # [b, e]
                    ori_2nd_ebd = tf.reduce_mean(support_ori_ebd_2nd, 1)  # [b, e]
                    ori_1st_ebd = tf.reduce_mean(support_ori_ebd_1st, 1)  # [b, e]

                    # perform aggregation
                    # 3order-2order
                    aggregate_3rd = tf.concat([support_encode_3rd, ori_3rd_ebd, ori_2nd_ebd],
                                              1)  # [b, 3e] third-order-neighbor || meta || second-order-neighbor
                    refined_second_neigh_ebd = tf.matmul(aggregate_3rd, w_2u)  # [b, 3e] * [3, e] = [b, e]

                    # 2order-1order
                    aggregate_2nd = tf.concat([refined_second_neigh_ebd, support_encode_2nd, ori_1st_ebd],
                                              1)  # [b, 3e] second-order-neighbor || meta || first-order-neighbor
                    refined_first_neigh_ebd = tf.matmul(aggregate_2nd, w_1u)  # [b, 3e] * [3e, e] = [b, e]
                    refined_target_ebd = tf.matmul(refined_first_neigh_ebd, w_0u)

                    intra_info = refined_target_ebd  # b, e

                    self.inter_support_3rd_user = tf.placeholder(tf.int32,
                                                                 shape=[None,
                                                                        None])  # 1st-order items and 3rd-order items
                    support_inter_ori_ebd_3rd = tf.nn.embedding_lookup(self.context_embedding_u,
                                                                       self.inter_support_3rd_user)
                    support_encode_inter_ebd_3rd = tf.reduce_mean(
                        self._encode(support_inter_ori_ebd_3rd, training=self.training_phrase_user_task), 1)  # [b, e]
                    inter_info = support_encode_inter_ebd_3rd

                    predict_u_3rd = tf.matmul(tf.concat([intra_info, inter_info], 1),
                                              w_concate)  # [b, 2e] * [2e, e] = [b, e]

                    batch_loss_3rd_user = Cosine_similarity(predict_u_3rd, self.target_user)
                    loss_3rd_user = -tf.reduce_mean(Cosine_similarity(predict_u_3rd, self.target_user))
                    optimizer_3rd_user_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_3rd_user)

            return predict_u_3rd, batch_loss_3rd_user, loss_3rd_user, optimizer_3rd_user_task

        elif name == 'LightGCN':
            print('use LightGCN')
            with tf.name_scope('LightGCN_3rd_u'):
                with tf.variable_scope('LightGCN_3rd_u', reuse=tf.AUTO_REUSE):
                    w_0u = self.glorot([2 * self.embedding_size, self.embedding_size], name='self_weights_0u')
                    # pre-process matrix
                    # AX item
                    pre_train_item_ebd = np.load(self.pre_train_i_ax)
                    c1 = tf.Variable(tf.constant(pre_train_item_ebd, shape=[self.num_items, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2ax')
                    self.context_embedding_i_normalized_1st = tf.concat([c1, c2], 0, name='embedding_item_ax')

                    # AX user
                    pre_train_user_ebd = np.load(self.pre_train_u_ax)
                    c1 = tf.Variable(tf.constant(pre_train_user_ebd, shape=[self.num_users, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2ax')
                    self.context_embedding_u_normalized_1st = tf.concat([c1, c2], 0, name='embedding_user_ax')

                    # AAX item
                    pre_train_item_ebd = np.load(self.pre_train_i_aax)
                    c1 = tf.Variable(tf.constant(pre_train_item_ebd, shape=[self.num_items, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2aax')
                    self.context_embedding_i_normalized_2nd = tf.concat([c1, c2], 0, name='embedding_item_aax')

                    # AAX user
                    pre_train_user_ebd = np.load(self.pre_train_u_aax)
                    c1 = tf.Variable(tf.constant(pre_train_user_ebd, shape=[self.num_users, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2aax')
                    self.context_embedding_u_normalized_2nd = tf.concat([c1, c2], 0, name='embedding_user_aax')

                    # AAAX user
                    pre_train_user_ebd = np.load(self.pre_train_u_aaax)
                    c1 = tf.Variable(tf.constant(pre_train_user_ebd, shape=[self.num_users, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2aaax')
                    self.context_embedding_u_normalized_3rd = tf.concat([c1, c2], 0, name='embedding_user_aaax')

                    # AAAX item
                    pre_train_item_ebd = np.load(self.pre_train_i_aaax)
                    c1 = tf.Variable(tf.constant(pre_train_item_ebd, shape=[self.num_items, self.embedding_size]),
                                     trainable=True)
                    c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2aaax')
                    self.context_embedding_i_normalized_3rd = tf.concat([c1, c2], 0, name='embedding_item_aaax')

                    # self.support_item_3rd = tf.placeholder(tf.int32, shape=[None, None])
                    support_ori_ebd_3rd = tf.nn.embedding_lookup(self.context_embedding_i_normalized_3rd,
                                                                 support_item_3rd)
                    ori_3rd_ebd = tf.reduce_mean(support_ori_ebd_3rd, 1)

                    # self.support_user_2nd_ = tf.placeholder(tf.int32, shape=[None, None])
                    support_ori_ebd_2nd = tf.nn.embedding_lookup(self.context_embedding_u_normalized_2nd,
                                                                 support_user_2nd_)
                    ori_2nd_ebd = tf.reduce_mean(support_ori_ebd_2nd, 1)

                    # self.support_item_1st_ = tf.placeholder(tf.int32, shape=[None, None])
                    support_ori_ebd_1st = tf.nn.embedding_lookup(self.context_embedding_i_normalized_1st,
                                                                 support_item_1st_)
                    ori_1st_ebd = tf.reduce_mean(support_ori_ebd_1st, 1)

                    aggregate_3rd = tf.concat(
                        [tf.reduce_sum(self._encode(support_ori_ebd_3rd, training=self.training_phrase_user_task), 1),
                         ori_3rd_ebd], 1)
                    aggregate_2nd = tf.concat(
                        [tf.reduce_sum(self._encode(support_ori_ebd_2nd, training=self.training_phrase_user_task), 1),
                         ori_2nd_ebd], 1)
                    aggregate_1st = tf.concat(
                        [tf.reduce_sum(self._encode(support_ori_ebd_1st, training=self.training_phrase_user_task), 1),
                         ori_1st_ebd], 1)

                    refined_target_ebd = tf.matmul(aggregate_3rd + aggregate_2nd + aggregate_1st, w_0u)

                    intra_info = refined_target_ebd
                    predict_u_3rd = intra_info
                    batch_loss_3rd_user = Cosine_similarity(predict_u_3rd, self.target_user)
                    loss_3rd_user = -tf.reduce_mean(Cosine_similarity(predict_u_3rd, self.target_user))
                    optimizer_3rd_user_task = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                                        initial_accumulator_value=1e-8).minimize(
                        loss_3rd_user)

            return predict_u_3rd, batch_loss_3rd_user, loss_3rd_user, optimizer_3rd_user_task


    #########  restore operation user ############
    def produce_reward_user_task(self, _target_user, b_k_shot_item, selected_input_2nd, b_oracle_user_ebd):
        b_second_order_users = selected_input_2nd  # revised second order users
        first_and_second_order_reward2, batch_predict_ebd, batch_target_ebd = self.sess.run(
            [self.batch_loss_2nd_user, self.predict_u_2nd, self.target_user],
            feed_dict={self.target_user: b_oracle_user_ebd,
                       self.support_item_1st: b_k_shot_item,
                       self.training_phrase_user_task: False,
                       self.support_user_2nd: b_second_order_users,
                       self.training_phrase_item_task: False})
        first_and_second_order_reward2 = np.reshape(first_and_second_order_reward2, (-1))
        batch_pearson = Pearson_correlation(batch_predict_ebd, batch_target_ebd)

        return first_and_second_order_reward2, batch_pearson

    def produce_reward_3rd_user_task(self, _target_user, b_k_shot_item, selected_input_2nd, selected_input_3rd,
                                     b_oracle_user_ebd):
        b_second_order_users = selected_input_2nd  # revised second order users
        b_third_order_items = selected_input_3rd  # revised third order items

        feed_dict = {self.target_user: b_oracle_user_ebd, self.support_item_1st_: b_k_shot_item,
                     self.training_phrase_user_task: False, self.support_user_2nd_: b_second_order_users,
                     self.training_phrase_item_task: False,
                     self.support_item_3rd: b_third_order_items}
        batch_evaluate_loss, batch_predict_ebd, batch_target_ebd = self.sess.run(
            [self.batch_loss_3rd_user, self.predict_u_3rd, self.target_user], feed_dict)
        batch_reward = np.reshape(batch_evaluate_loss, (-1))
        batch_pearson = Pearson_correlation(batch_predict_ebd, batch_target_ebd)
        return batch_reward, batch_pearson

    def produce_original_reward_user_task(self, train_batches, num_batch_train, ClassData):
        original_reward = 0.0
        for batch_index in range(num_batch_train):
            b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item = ClassData.batch_gen_3rd_user_task(
                train_batches, batch_index)
            batch_reward = -self.sess.run(self.loss_2nd_user,
                                          feed_dict={self.target_user: b_oracle_user_ebd,
                                                     self.support_item_1st: b_k_shot_item,
                                                     self.training_phrase_user_task: False,
                                                     self.support_user_2nd: b_second_order_users,
                                                     self.training_phrase_item_task: False})
            original_reward += batch_reward
        return original_reward / num_batch_train

    def produce_original_reward_3rd_user_task(self, train_batches, num_batch_train, ClassData):
        original_reward = 0.0
        for batch_index in range(num_batch_train):
            b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item = ClassData.batch_gen_3rd_user_task(
                train_batches, batch_index)
            feed_dict = {self.target_user: b_oracle_user_ebd, self.support_item_1st_: b_k_shot_item,
                         self.training_phrase_user_task: False,
                         self.support_user_2nd_: b_second_order_users,
                         self.training_phrase_item_task: False,
                         self.support_item_3rd: b_third_order_items}
            batch_reward = -self.sess.run(self.loss_3rd_user,
                                          feed_dict=feed_dict)
            original_reward += batch_reward
        return original_reward / num_batch_train

    def generate_batch_state_ebd_user_task(self, train_batches, ClassData):
        all_target_ori_ebd, all_target_aggre_ebd, all_target_second_aggre_ebd, all_first_neigh_ori_ebd, all_second_neigh_ori_ebd, all_third_neigh_ori_ebd, all_first_order_reward1, all_first_and_second_order_reward2, all_former_third_order_reward3 = [], [], [], [], [], [], [], [], []
        num_batch_train = ClassData.oracle_num_users // setting.batch_size
        train_batch_index = range(num_batch_train)
        for index in train_batch_index:
            b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item = ClassData.batch_gen_3rd_user_task(
                train_batches, index)


            target_ori_ebd = self.original_user_ebd[b_target_user]

            target_aggre_ebd = self.sess.run(self.final_support_encode_user_task,
                                             feed_dict={self.support_item: b_k_shot_item,
                                                        self.training_phrase_user_task: False})
            target_second_aggre_ebd = self.sess.run(self.final_support_encode_item_task,
                                                    feed_dict={self.support_user: b_second_order_users,
                                                               self.training_phrase_item_task: False})

            first_order_reward1 = self.sess.run(self.batch_loss_user_task,
                                                feed_dict={self.support_item: b_k_shot_item,
                                                           self.target_user: b_oracle_user_ebd,
                                                           self.training_phrase_user_task: False})

            first_and_second_order_reward2 = self.sess.run(self.batch_loss_2nd_user,
                                                           feed_dict={self.target_user: b_oracle_user_ebd,
                                                                      self.support_item_1st: b_k_shot_item,
                                                                      self.training_phrase_user_task: False,
                                                                      self.support_user_2nd: b_second_order_users,
                                                                      self.training_phrase_item_task: False})

            former_third_order_reward3 = self.sess.run(self.batch_loss_3rd_user,
                                                       feed_dict={self.target_user: b_oracle_user_ebd,
                                                                  self.support_item_1st_: b_k_shot_item,
                                                                  self.training_phrase_user_task: False,
                                                                  self.support_user_2nd_: b_second_order_users,
                                                                  self.training_phrase_item_task: False,
                                                                  self.support_item_3rd: b_third_order_items})

            first_neigh_ori_ebd = self.original_item_ebd[b_k_shot_item]
            second_neigh_ori_ebd = self.original_user_ebd[b_second_order_users]
            third_neigh_ori_ebd = self.original_item_ebd[b_third_order_items]

            all_target_ori_ebd.append(target_ori_ebd)
            all_target_aggre_ebd.append(target_aggre_ebd)
            all_target_second_aggre_ebd.append(target_second_aggre_ebd)
            all_first_neigh_ori_ebd.append(first_neigh_ori_ebd)
            all_second_neigh_ori_ebd.append(second_neigh_ori_ebd)
            all_third_neigh_ori_ebd.append(third_neigh_ori_ebd)
            all_first_order_reward1.append(first_order_reward1)
            all_first_and_second_order_reward2.append(first_and_second_order_reward2)
            all_former_third_order_reward3.append(former_third_order_reward3)


        return all_target_ori_ebd, all_target_aggre_ebd, all_target_second_aggre_ebd, all_first_neigh_ori_ebd, all_second_neigh_ori_ebd, all_third_neigh_ori_ebd, all_first_order_reward1, all_first_and_second_order_reward2, all_former_third_order_reward3

    def evaluate_user_task(self):

        data_valid = data.Dataset(setting.oracle_valid_file_user_task)
        valid_batches = data_valid.get_positive_instances_user_task(random_seed=0)
        num_batch_valid = data_valid.oracle_num_users // setting.batch_size
        valid_batch_index = range(num_batch_valid)

        evaluate_loss, evaluate_pearson = 0.0, 0.0
        for index in tqdm.tqdm(valid_batch_index):
            b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item = data_valid.batch_gen_3rd_user_task(
                valid_batches, index)
            feed_dict = {self.target_user: b_oracle_user_ebd, self.support_item: b_k_shot_item,
                         self.training_phrase_user_task: False,
                         self.support_user: b_second_order_users,
                         self.training_phrase_item_task: False,
                         self.support_user_2nd: b_second_order_users}

            batch_evaluate_loss, batch_predict_ebd, batch_target_ebd = self.sess.run(
                [self.loss_2nd_user, self.predict_u_2nd, self.target_user], feed_dict)
            evaluate_loss += batch_evaluate_loss

            batch_pearson = Pearson_correlation(batch_predict_ebd, batch_target_ebd)
            evaluate_pearson += batch_pearson
        return evaluate_loss / len(valid_batch_index), evaluate_pearson / len(valid_batch_index)

    def evaluate_3rd_user_task(self):
        data_valid = data.Dataset(setting.oracle_valid_file_user_task)
        valid_batches = data_valid.get_positive_instances_user_task(random_seed=0)
        num_batch_valid = data_valid.oracle_num_users // setting.batch_size
        valid_batch_index = range(num_batch_valid)

        evaluate_loss, evaluate_pearson = 0.0, 0.0
        for index in tqdm.tqdm(valid_batch_index):
            b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item = data_valid.batch_gen_3rd_user_task(
                valid_batches, index)
            feed_dict = {self.target_user: b_oracle_user_ebd, self.support_item_1st_: b_k_shot_item,
                         self.training_phrase_user_task: False,
                         self.support_user_2nd_: b_second_order_users,
                         self.training_phrase_item_task: False,
                         self.support_item_3rd: b_third_order_items}

            batch_evaluate_loss, batch_predict_ebd, batch_target_ebd = self.sess.run(
                [self.loss_3rd_user, self.predict_u_3rd, self.target_user], feed_dict)
            batch_evaluate_loss = -batch_evaluate_loss
            evaluate_loss += batch_evaluate_loss

            batch_pearson = Pearson_correlation(batch_predict_ebd, batch_target_ebd)
            evaluate_pearson += batch_pearson
        return evaluate_loss / len(valid_batch_index), evaluate_pearson / len(valid_batch_index)

    ######## restore item task #######

    def produce_reward_item_task(self, _target_item, b_k_shot_user, selected_input, b_oracle_item_ebd):
        b_second_order_items = selected_input  # revised second order users
        first_and_second_order_reward2, batch_predict_ebd, batch_target_ebd = self.sess.run(
            [self.batch_loss_2nd_item_pos, self.predict_i_2nd_pos, self.target_item],
            feed_dict={self.target_item: b_oracle_item_ebd,
                       self.support_user_1st_pos: b_k_shot_user,
                       self.training_phrase_user_task: False,
                       self.support_item_2nd_pos: b_second_order_items,
                       self.training_phrase_item_task: False})
        first_and_second_order_reward2 = np.reshape(first_and_second_order_reward2, (-1))
        batch_pearson = Pearson_correlation(batch_predict_ebd, batch_target_ebd)

        return first_and_second_order_reward2, batch_pearson

    def produce_reward_3rd_item_task(self, _target_item, b_k_shot_user, selected_input_2nd, selected_input_3rd,
                                     b_oracle_item_ebd):
        b_second_order_items = selected_input_2nd  # revised second order users
        b_third_order_users = selected_input_3rd  # revised third order items

        feed_dict = {self.target_item: b_oracle_item_ebd, self.support_user_1st_pos_: b_k_shot_user,
                     self.training_phrase_user_task: False, self.support_item_2nd_pos_: b_second_order_items,
                     self.training_phrase_item_task: False,
                     self.support_user_3rd_pos: b_third_order_users}
        batch_evaluate_loss, batch_predict_ebd, batch_target_ebd = self.sess.run(
            [self.batch_loss_3rd_item_pos, self.predict_i_3rd_pos, self.target_item], feed_dict)
        batch_reward = np.reshape(batch_evaluate_loss, (-1))
        batch_pearson = Pearson_correlation(batch_predict_ebd, batch_target_ebd)
        return batch_reward, batch_pearson

    def produce_original_reward_item_task(self, train_batches, num_batch_train, ClassData):
        original_reward = 0.0
        for batch_index in range(num_batch_train):
            b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user = ClassData.batch_gen_3rd_item_task(
                train_batches, batch_index)
            batch_reward = -self.sess.run(self.loss_2nd_item_pos,
                                          feed_dict={self.target_item: b_oracle_item_ebd,
                                                     self.support_user_1st_pos: b_k_shot_user,
                                                     self.training_phrase_user_task: False,
                                                     self.support_item_2nd_pos: b_second_order_items,
                                                     self.training_phrase_item_task: False})
            original_reward += batch_reward
        return original_reward / num_batch_train

    def produce_original_reward_3rd_item_task(self, train_batches, num_batch_train, ClassData):
        original_reward = 0.0
        for batch_index in range(num_batch_train):
            b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user = ClassData.batch_gen_3rd_item_task(
                train_batches, batch_index)
            feed_dict = {self.target_item: b_oracle_item_ebd, self.support_user_1st_pos_: b_k_shot_user,
                         self.training_phrase_user_task: False,
                         self.support_item_2nd_pos_: b_second_order_items,
                         self.training_phrase_item_task: False,
                         self.support_user_3rd_pos: b_third_order_users}
            batch_reward = -self.sess.run(self.loss_3rd_item_pos,
                                          feed_dict=feed_dict)
            original_reward += batch_reward
        return original_reward / num_batch_train

    def generate_batch_state_ebd_item_task(self, train_batches, ClassData):
        all_target_ori_ebd, all_target_aggre_ebd, all_target_second_aggre_ebd, all_first_neigh_ori_ebd, all_second_neigh_ori_ebd, all_third_neigh_ori_ebd, all_first_order_reward1, all_first_and_second_order_reward2, all_former_third_order_reward3 = [], [], [], [], [], [], [], [], []
        num_batch_train = ClassData.oracle_num_items // setting.batch_size
        train_batch_index = range(num_batch_train)
        for index in train_batch_index:
            b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user = ClassData.batch_gen_3rd_item_task(
                train_batches, index)


            target_ori_ebd = self.original_item_ebd[b_target_item]
            target_aggre_ebd = self.sess.run(self.final_support_encode_item_task,
                                             feed_dict={self.support_user: b_k_shot_user,
                                                        self.training_phrase_item_task: False})

            target_second_aggre_ebd = self.sess.run(self.final_support_encode_user_task,
                                                    feed_dict={self.support_item: b_second_order_items,
                                                               self.training_phrase_user_task: False})

            first_order_reward1 = self.sess.run(self.batch_loss_item_task,
                                                feed_dict={self.support_user: b_k_shot_user,
                                                           self.target_item: b_oracle_item_ebd,
                                                           self.training_phrase_item_task: False})

            first_and_second_order_reward2 = self.sess.run(self.batch_loss_2nd_item_pos,
                                                           feed_dict={self.target_item: b_oracle_item_ebd,
                                                                      self.support_user_1st_pos: b_k_shot_user,
                                                                      self.training_phrase_user_task: False,
                                                                      self.support_item_2nd_pos: b_second_order_items,
                                                                      self.training_phrase_item_task: False})
            former_third_order_reward3 = self.sess.run(self.batch_loss_3rd_item_pos,
                                                       feed_dict={self.target_item: b_oracle_item_ebd,
                                                                  self.support_user_1st_pos_: b_k_shot_user,
                                                                  self.training_phrase_user_task: False,
                                                                  self.support_item_2nd_pos_: b_second_order_items,
                                                                  self.training_phrase_item_task: False,
                                                                  self.support_user_3rd_pos: b_third_order_users})

            first_neigh_ori_ebd = self.original_user_ebd[b_k_shot_user]
            second_neigh_ori_ebd = self.original_item_ebd[b_second_order_items]
            third_neigh_ori_ebd = self.original_user_ebd[b_third_order_users]

            all_target_ori_ebd.append(target_ori_ebd)
            all_target_aggre_ebd.append(target_aggre_ebd)
            all_target_second_aggre_ebd.append(target_second_aggre_ebd)
            all_first_neigh_ori_ebd.append(first_neigh_ori_ebd)
            all_second_neigh_ori_ebd.append(second_neigh_ori_ebd)
            all_third_neigh_ori_ebd.append(third_neigh_ori_ebd)
            all_first_order_reward1.append(first_order_reward1)
            all_first_and_second_order_reward2.append(first_and_second_order_reward2)
            all_former_third_order_reward3.append(former_third_order_reward3)


        return all_target_ori_ebd, all_target_aggre_ebd, all_target_second_aggre_ebd, all_first_neigh_ori_ebd, all_second_neigh_ori_ebd, all_third_neigh_ori_ebd, all_first_order_reward1, all_first_and_second_order_reward2, all_former_third_order_reward3

    def evaluate_item_task(self):
        data_valid = data.Dataset(setting.oracle_valid_file_item_task)
        valid_batches = data_valid.get_positive_instances_item_task(random_seed=0)
        num_batch_valid = data_valid.oracle_num_items // setting.batch_size
        valid_batch_index = range(num_batch_valid)

        evaluate_loss, evaluate_pearson = 0.0, 0.0
        for index in tqdm.tqdm(valid_batch_index):
            b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user = data_valid.batch_gen_3rd_item_task(
                valid_batches, index)

            feed_dict = {self.target_item: b_oracle_item_ebd, self.support_user_1st_pos: b_k_shot_user,
                         self.training_phrase_user_task: False,
                         self.training_phrase_item_task: False,
                         self.support_item_2nd_pos: b_second_order_items}

            batch_evaluate_loss, batch_predict_ebd, batch_target_ebd = self.sess.run(
                [self.loss_2nd_item_pos, self.predict_i_2nd_pos, self.target_item], feed_dict)
            batch_evaluate_loss = -batch_evaluate_loss
            evaluate_loss += batch_evaluate_loss

            batch_pearson = Pearson_correlation(batch_predict_ebd, batch_target_ebd)
            evaluate_pearson += batch_pearson
        return evaluate_loss / len(valid_batch_index), evaluate_pearson / len(valid_batch_index)

    def evaluate_3rd_item_task(self):
        data_valid = data.Dataset(setting.oracle_valid_file_item_task)
        valid_batches = data_valid.get_positive_instances_item_task(random_seed=0)
        num_batch_valid = data_valid.oracle_num_items // setting.batch_size
        valid_batch_index = range(num_batch_valid)

        evaluate_loss, evaluate_pearson = 0.0, 0.0
        for index in tqdm.tqdm(valid_batch_index):
            b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user = data_valid.batch_gen_3rd_item_task(
                valid_batches, index)
            feed_dict = {self.target_item: b_oracle_item_ebd, self.support_user_1st_pos_: b_k_shot_user,
                         self.training_phrase_user_task: False,
                         self.support_item_2nd_pos_: b_second_order_items,
                         self.training_phrase_item_task: False,
                         self.support_user_3rd_pos: b_third_order_users}

            batch_evaluate_loss, batch_predict_ebd, batch_target_ebd = self.sess.run(
                [self.loss_3rd_item_pos, self.predict_i_3rd_pos, self.target_item], feed_dict)
            batch_evaluate_loss = -batch_evaluate_loss
            evaluate_loss += batch_evaluate_loss

            batch_pearson = Pearson_correlation(batch_predict_ebd, batch_target_ebd)
            evaluate_pearson += batch_pearson
        return evaluate_loss / len(valid_batch_index), evaluate_pearson / len(valid_batch_index)

    def generate_batch_state_ebd_item_task_downstream(self, train_batches, ClassData):
        all_target_ori_ebd, all_target_aggre_ebd, all_target_second_aggre_ebd, all_first_neigh_ori_ebd, all_second_neigh_ori_ebd, all_third_neigh_ori_ebd, all_first_order_reward1, all_first_and_second_order_reward2, all_former_third_order_reward3 = [], [], [], [], [], [], [], [], []
        num_batch_train = ClassData.oracle_num_items // setting.batch_size
        train_batch_index = range(num_batch_train)
        for index in train_batch_index:
            b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user = ClassData.batch_gen_mix_item_task(
                train_batches, index)

            b_target_item = np.reshape(b_target_item, (-1))

            target_ori_ebd = self.original_item_ebd[b_target_item]
            target_aggre_ebd = self.sess.run(self.final_support_encode_item_task,
                                             feed_dict={self.support_user: b_k_shot_user,
                                                        self.training_phrase_item_task: False})

            target_second_aggre_ebd = self.sess.run(self.final_support_encode_user_task,
                                                    feed_dict={self.support_item: b_second_order_items,
                                                               self.training_phrase_user_task: False})

            first_order_reward1 = self.sess.run(self.batch_loss_item_task,
                                                feed_dict={self.support_user: b_k_shot_user,
                                                           self.target_item: b_oracle_item_ebd,
                                                           self.training_phrase_item_task: False})

            first_and_second_order_reward2 = self.sess.run(self.batch_loss_2nd_item_pos,
                                                           feed_dict={self.target_item: b_oracle_item_ebd,
                                                                      self.support_user_1st_pos: b_k_shot_user,
                                                                      self.training_phrase_user_task: False,
                                                                      self.support_item_2nd_pos: b_second_order_items,
                                                                      self.training_phrase_item_task: False})
            former_third_order_reward3 = self.sess.run(self.batch_loss_3rd_item_pos,
                                                       feed_dict={self.target_item: b_oracle_item_ebd,
                                                                  self.support_user_1st_pos_: b_k_shot_user,
                                                                  self.training_phrase_user_task: False,
                                                                  self.support_item_2nd_pos_: b_second_order_items,
                                                                  self.training_phrase_item_task: False,
                                                                  self.support_user_3rd_pos: b_third_order_users})

            first_neigh_ori_ebd = self.original_user_ebd[b_k_shot_user]
            second_neigh_ori_ebd = self.original_item_ebd[b_second_order_items]
            third_neigh_ori_ebd = self.original_user_ebd[b_third_order_users]

            all_target_ori_ebd.append(target_ori_ebd)
            all_target_aggre_ebd.append(target_aggre_ebd)
            all_target_second_aggre_ebd.append(target_second_aggre_ebd)
            all_first_neigh_ori_ebd.append(first_neigh_ori_ebd)
            all_second_neigh_ori_ebd.append(second_neigh_ori_ebd)
            all_third_neigh_ori_ebd.append(third_neigh_ori_ebd)
            all_first_order_reward1.append(first_order_reward1)
            all_first_and_second_order_reward2.append(first_and_second_order_reward2)
            all_former_third_order_reward3.append(former_third_order_reward3)

        return all_target_ori_ebd, all_target_aggre_ebd, all_target_second_aggre_ebd, all_first_neigh_ori_ebd, all_second_neigh_ori_ebd, all_third_neigh_ori_ebd, all_first_order_reward1, all_first_and_second_order_reward2, all_former_third_order_reward3

    def generate_batch_state_ebd_user_task_downstream(self, train_batches, ClassData):
        all_target_ori_ebd, all_target_aggre_ebd, all_target_second_aggre_ebd, all_first_neigh_ori_ebd, all_second_neigh_ori_ebd, all_third_neigh_ori_ebd, all_first_order_reward1, all_first_and_second_order_reward2, all_former_third_order_reward3 = [], [], [], [], [], [], [], [], []
        num_batch_train = ClassData.oracle_num_users // setting.batch_size

        train_batch_index = range(num_batch_train)
        for index in train_batch_index:
            b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item = ClassData.batch_gen_mix_user_task(
                train_batches, index)

            b_target_user = np.reshape(b_target_user, (-1))
            b_mask_num_second_order_user = np.reshape(b_mask_num_second_order_user, (-1))


            target_ori_ebd = self.original_user_ebd[b_target_user]

            target_aggre_ebd = self.sess.run(self.final_support_encode_user_task,
                                             feed_dict={self.support_item: b_k_shot_item,
                                                        self.training_phrase_user_task: False})
            target_second_aggre_ebd = self.sess.run(self.final_support_encode_item_task,
                                                    feed_dict={self.support_user: b_second_order_users,
                                                               self.training_phrase_item_task: False})
            first_order_reward1 = self.sess.run(self.batch_loss_user_task,
                                                feed_dict={self.support_item: b_k_shot_item,
                                                           self.target_user: b_oracle_user_ebd,
                                                           self.training_phrase_user_task: False})

            first_and_second_order_reward2 = self.sess.run(self.batch_loss_2nd_user,
                                                           feed_dict={self.target_user: b_oracle_user_ebd,
                                                                      self.support_item_1st: b_k_shot_item,
                                                                      self.training_phrase_user_task: False,
                                                                      self.support_user_2nd: b_second_order_users,
                                                                      self.training_phrase_item_task: False})
            former_third_order_reward3 = self.sess.run(self.batch_loss_3rd_user,
                                                       feed_dict={self.target_user: b_oracle_user_ebd,
                                                                  self.support_item_1st_: b_k_shot_item,
                                                                  self.training_phrase_user_task: False,
                                                                  self.support_user_2nd_: b_second_order_users,
                                                                  self.training_phrase_item_task: False,
                                                                  self.support_item_3rd: b_third_order_items})

            first_neigh_ori_ebd = self.original_item_ebd[b_k_shot_item]
            second_neigh_ori_ebd = self.original_user_ebd[b_second_order_users]
            third_neigh_ori_ebd = self.original_item_ebd[b_third_order_items]


            all_target_ori_ebd.append(target_ori_ebd)
            all_target_aggre_ebd.append(target_aggre_ebd)  
            all_target_second_aggre_ebd.append(target_second_aggre_ebd)
            all_first_neigh_ori_ebd.append(first_neigh_ori_ebd)
            all_second_neigh_ori_ebd.append(second_neigh_ori_ebd)
            all_third_neigh_ori_ebd.append(third_neigh_ori_ebd)
            all_first_order_reward1.append(first_order_reward1)
            all_first_and_second_order_reward2.append(first_and_second_order_reward2)
            all_former_third_order_reward3.append(former_third_order_reward3)

        return all_target_ori_ebd, all_target_aggre_ebd, all_target_second_aggre_ebd, all_first_neigh_ori_ebd, all_second_neigh_ori_ebd, all_third_neigh_ori_ebd, all_first_order_reward1, all_first_and_second_order_reward2, all_former_third_order_reward3

    def produce_original_reward_item_task_downstream(self, train_batches, num_batch_train, ClassData):
        original_reward = 0.0
        for batch_index in range(num_batch_train):
            b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user = ClassData.batch_gen_mix_item_task(
                train_batches, batch_index)
            batch_reward = -self.sess.run(self.loss_2nd_item_pos,
                                          feed_dict={self.target_item: b_oracle_item_ebd,
                                                     self.support_user_1st_pos: b_k_shot_user,
                                                     self.training_phrase_user_task: False,
                                                     self.support_item_2nd_pos: b_second_order_items,
                                                     self.training_phrase_item_task: False})
            original_reward += batch_reward
        return original_reward / num_batch_train

    def produce_original_reward_3rd_item_task_downstream(self, train_batches, num_batch_train, ClassData):
        original_reward = 0.0
        for batch_index in range(num_batch_train):
            b_target_item, b_k_shot_user, b_second_order_items, b_third_order_users, b_oracle_item_ebd, b_mask_num_second_order_item, b_mask_num_third_order_user = ClassData.batch_gen_mix_item_task(
                train_batches, batch_index)
            feed_dict = {self.target_item: b_oracle_item_ebd, self.support_user_1st_pos_: b_k_shot_user,
                         self.training_phrase_user_task: False,
                         self.support_item_2nd_pos_: b_second_order_items,
                         self.training_phrase_item_task: False,
                         self.support_user_3rd_pos: b_third_order_users}
            batch_reward = -self.sess.run(self.loss_3rd_item_pos,
                                          feed_dict=feed_dict)
            original_reward += batch_reward
        return original_reward / num_batch_train

    def produce_original_reward_user_task_downstream(self, train_batches, num_batch_train, ClassData):
        original_reward = 0.0
        for batch_index in range(num_batch_train):
            b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item = ClassData.batch_gen_mix_user_task(
                train_batches, batch_index)
            batch_reward = -self.sess.run(self.loss_2nd_user,
                                          feed_dict={self.target_user: b_oracle_user_ebd,
                                                     self.support_item_1st: b_k_shot_item,
                                                     self.training_phrase_user_task: False,
                                                     self.support_user_2nd: b_second_order_users,
                                                     self.training_phrase_item_task: False})
            original_reward += batch_reward
        return original_reward / num_batch_train

    def produce_original_reward_3rd_user_task_downstream(self, train_batches, num_batch_train, ClassData):
        original_reward = 0.0
        for batch_index in range(num_batch_train):
            b_target_user, b_k_shot_item, b_second_order_users, b_third_order_items, b_oracle_user_ebd, b_mask_num_second_order_user, b_mask_num_third_order_item = ClassData.batch_gen_mix_user_task(
                train_batches, batch_index)
            feed_dict = {self.target_user: b_oracle_user_ebd, self.support_item_1st_: b_k_shot_item,
                         self.training_phrase_user_task: False,
                         self.support_user_2nd_: b_second_order_users,
                         self.training_phrase_item_task: False,
                         self.support_item_3rd: b_third_order_items}
            batch_reward = -self.sess.run(self.loss_3rd_user,
                                          feed_dict=feed_dict)
            original_reward += batch_reward
        return original_reward / num_batch_train


if __name__ == '__main__':
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        name = 'GraphSAGE'
        # name = 'GAT'
        # name = 'LightGCN'
 

        if name == 'GraphSAGE':
            model = GeneralGNN('GraphSAGE', sess)
            sess.run(tf.global_variables_initializer())
            logging.info("initialized")
            print('training user task...')
            training_user_task(model, sess)
            training_2nd_user_task(model, sess)
            training_3rd_user_task(model, sess)
            train_user_rl_task('GraphSAGE')
            revise_rl_user_task('GraphSAGE')
            print('training item task...')
            training_item_task(model, sess)
            training_2nd_item_task(model, sess)
            training_3rd_item_task(model, sess)
            train_item_rl_task('GraphSAGE')
            revise_rl_item_task('GraphSAGE')

        if name == 'GAT':
            model = GeneralGNN('GAT', sess)
            sess.run(tf.global_variables_initializer())
            logging.info("initialized")
            print('training user task...')
            training_user_task(model, sess)
            training_2nd_user_task(model, sess)
            training_3rd_user_task(model, sess)
            train_user_rl_task('GAT')
            revise_rl_user_task('GAT')
            print('training item task...')
            training_item_task(model, sess)
            training_2nd_item_task(model, sess)
            training_3rd_item_task(model, sess)
            train_item_rl_task('GAT')
            revise_rl_item_task('GAT')


        if name == 'LightGCN':
            model = GeneralGNN('LightGCN', sess)
            sess.run(tf.global_variables_initializer())
            logging.info("initialized")
            print('training user task...')
            training_user_task(model, sess)
            training_2nd_user_task(model, sess)
            training_3rd_user_task(model, sess)
            train_user_rl_task('LightGCN')
            revise_rl_user_task('LightGCN')
            print('training item task...')
            training_item_task(model, sess)
            training_2nd_item_task(model, sess)
            training_3rd_item_task(model, sess)
            train_item_rl_task('LightGCN')
            revise_rl_item_task('LightGCN')

   

