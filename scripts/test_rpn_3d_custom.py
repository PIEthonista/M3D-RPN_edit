# -----------------------------------------
# python modules
# -----------------------------------------
from importlib import import_module
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
import sys
import numpy as np
import os

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# custom modules
# -----------------------------------------
from lib.imdb_util import *

conf_path = 'config/m3d_rpn_depth_aware_test_config.pkl'
weights_path = 'weights/M3D-RPN-Release/m3d_rpn_depth_aware_test'

# =============
dataset_test_img_path = '/work/u5832291/view_neti_RT/data/DistentangledCarlaScenes/images/minicooper_day_both'
dataset_test_calib_path = '/work/u5832291/view_neti_RT/data/DistentangledCarlaScenes/pose_matrices_RT_only'
output_results_hill_climbed = False

results_path = 'output/DistentangledCarlaScenes/minicooper_day_both/data'

# load config
conf = edict(pickle_read(conf_path))
conf.pretrained = None

data_path = os.path.join(os.getcwd(), 'data')
# results_path = os.path.join('output', 'tmp_results', 'data')

# make directory
# mkdir_if_missing(results_path, delete_if_exist=True)

# -----------------------------------------
# torch defaults
# -----------------------------------------

# defaults
init_torch(conf.rng_seed, conf.cuda_seed)

# -----------------------------------------
# setup network
# -----------------------------------------

# net
net = import_module('models.' + conf.model).build(conf)

# load weights
load_weights(net, weights_path, remove_module=True)

# switch modes for evaluation
net.eval()

print(pretty_print('conf', conf))

# -----------------------------------------
# test kitti
# -----------------------------------------

# test_kitti_3d(conf.dataset_test, net, conf, results_path, data_path, use_log=False)
test_kitti_3d_custom(dataset_test_img_path, dataset_test_calib_path, net, conf, results_path, use_log=False, output_results_hill_climbed=output_results_hill_climbed)