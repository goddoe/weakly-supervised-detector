from libs.tiny_imagenet_utils import read_tiny_imagenet, save_with_tfrecord
from libs.various_utils import save_as_pickle
from configs.project_config import project_path


# ======================================
# Paths
tiny_imagenet_dir_path = "{}/data/tiny_imagenet_200".format(project_path)

tfrecord_train_dir = "{}/data/tiny_imagenet_200/tfrecord/train".format(project_path)
tfrecord_valid_dir = "{}/data/tiny_imagenet_200/tfrecord/valid".format(project_path)
tfrecord_test_dir = "{}/data/tiny_imagenet_200/tfrecord/test".format(project_path)
pickle_save_path = "{}/data/tiny_imagenet_200/pickle/tiny_imagenet.pickle".format(project_path)


# ======================================
# Save 
d = read_tiny_imagenet(tiny_imagenet_dir_path, train_ratio=0.8)

save_with_tfrecord(tfrecord_train_dir,
                   X=d['X_train'],
                   Y=d['Y_train'],
                   Y_one_hot=d['Y_train_one_hot'],
                   P=d['P_train'],
                   label_depth=200,
                   shard_size=2000,
                   prefix='train')


save_with_tfrecord(tfrecord_valid_dir,
                   X=d['X_valid'],
                   Y=d['Y_valid'],
                   Y_one_hot=d['Y_valid_one_hot'],
                   P=d['P_valid'],
                   label_depth=200,
                   shard_size=2000,
                   prefix='valid')


save_with_tfrecord(tfrecord_test_dir,
                   X=d['X_test'],
                   Y=d['Y_test'],
                   Y_one_hot=d['Y_test_one_hot'],
                   P=d['P_test'],
                   label_depth=200,
                   shard_size=2000,
                   prefix='test')


save_as_pickle(d, pickle_save_path, flag_make_path=True)
