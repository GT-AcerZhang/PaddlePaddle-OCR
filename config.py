# data dict
dict_path = "dataset/dict.txt"
# Data shape
data_shape = [1, 48, 512]
# Minibatch size.
batch_size = 128
# Learning rate.
lr = 1e-3
# Learning rate decay strategy. 'piecewise_decay' or None is valid.
lr_decay_strategy = None
# L2 decay rate.
l2decay = 4e-4
# Momentum rate.
momentum = 0.9
# The threshold of gradient clipping.
gradient_clip = 10.0
# The number of iterations.
total_step = 720000
# Log period.
log_period = 1000
# character class num.
num_classes = 95
# Save model period. '-1' means never saving the model.
save_model_period = 2000
# Evaluate period. '-1' means never evaluating the model.
eval_period = 15000
# The list file of images to be used for training.
train_list = 'dataset/train.txt'
# The list file of images to be used for training.
test_list = 'dataset/test.txt'
train_prefix = 'dataset/train'
test_prefix = 'dataset/test'
# Which type of network to be used. 'crnn_ctc' or 'attention'
use_model = 'crnn_ctc'
# Save model path
persistables_models_path = 'models/%s/persistables/' % use_model
infer_model_path = 'models/%s/infer/' % use_model
# The init model file of directory.
pretrained_model = None
# Whether use GPU to train.
use_gpu = True
# Min average window.
min_average_window = 10000
# Max average window. It is proposed to be set as the number of minibatch in a pass.
max_average_window = 12500
# Average window.
average_window = 0.15
# Whether use parallel training.
parallel = True

