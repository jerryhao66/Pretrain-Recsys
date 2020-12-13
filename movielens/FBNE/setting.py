batch_size = 32
epochs = 20
learning_rate = 0.003
aggregator_learning_rate = 0.01
learning_rate_downstream = 0.02 
num_users = 6041
num_items = 3953
negative_num = 99
checkpoint_path_user_task = '../Checkpoint/user_task/'
checkpoint_path_item_task = '../Checkpoint/item_task/'
hidden_dim = 256
user_epoch = 5
item_epoch = 25
verbose = 1
second_user_epoch = 10
third_user_epoch = 10
second_item_epoch = 10
third_item_epoch = 10

dropout_rate = 0
num_heads = 4
d_ff = 4
num_blocks = 2



support_num = 3
kshot_num = 3
kshot_second_num = 3
kshot_third_num = 3

checkpoint_path_rl_user_task = '../Checkpoint/rl_agent/user_task/'
checkpoint_path_rl_item_task = '../Checkpoint/rl_agent/item_task/'


oracle_user_ebd_path = '../ml_oracle_ebd/user_feature.npy'
oracle_item_ebd_path = '../ml_oracle_ebd/item_feature.npy'

pre_train_item_ebd_path = '../ml_oracle_ebd/item_feature.npy'
pre_train_user_ebd_path = '../ml_oracle_ebd/user_feature.npy'

original_user_ebd = '../ml_oracle_ebd/user_feature.npy'
original_item_ebd = '../ml_oracle_ebd/item_feature.npy'
embedding_size = 256
state_size = 2819
second_order_state_size = 2819
state_size_3rd = 2821
third_order_state_size = 2821
sample_times = 3



oracle_training_file_user_task = '../movielens-upstream/user_task/user_task_train_oracle_rating.csv'
oracle_valid_file_user_task = '../movielens-upstream/user_task/user_task_valid_oracle_rating.csv'

oracle_training_file_item_task = '../movielens-upstream/item_task/item_task_train_oracle_rating.csv'
oracle_valid_file_item_task = '../movielens-upstream/item_task/item_task_valid_oracle_rating.csv'

k = 30


agent_pretrain_tau = 0.5
agent_weight_size = 8
dropout = 0.95

fastgcn_pre_train_concat_ebd = './ml_oracle_ebd/ml_feature.npy'
support_size = 2

fastgcn_batch_size = 64


'''
lightgcn parameter
'''
pre_train_i_ax_path = '../ml_oracle_ebd/1st_i.npy'
pre_train_u_ax_path = '../ml_oracle_ebd/1st_u.npy'
pre_train_u_aax_path = '../ml_oracle_ebd/2nd_u.npy'
pre_train_i_aax_path = '../ml_oracle_ebd/2nd_i.npy'
pre_train_i_aaax_path = '../ml_oracle_ebd/3rd_i.npy'
pre_train_u_aaax_path = '../ml_oracle_ebd/3rd_u.npy'

##### downstream recommendation ######
Ks = [20]

downstream_meta_train_file = '../movielens-downstream/movielens/user_task_meta_train.csv' # include meta training users
downstream_support_file = '../movielens-downstream/movielens/support_1_shot.csv' # meta test uesrs support file
downstream_query_file = '../movielens-downstream/movielens/query_1_shot.csv'  # ground-truth
decay = 1e-5


