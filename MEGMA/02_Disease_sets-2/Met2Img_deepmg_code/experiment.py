"""
================================================================================
dev_met2img.py : Predicting diseases based on images (EMBEDDINGs) 
        with stratified cross-validation (including support building images and training)
================================================================================
Author: Thanh Hai Nguyen
(**this version is still developping)
date: 02-11-2018 (updated to 30/01/2020)
this file includes functions to support for an experiment and code related to use of parameters

1. reading parameters to run
    para_cmd: read parameters from command line
    get_default_value: set default value to para which is not set yet
    validation_para: validate para
    string_to_numeric: convert from string to numeric para which owns numeric type
    write_para: write para to config file    

2. functions to support for main function
    naming_folder: naming folder in training/learning process
    name_log_final: naming log file
    read_data: read data 
    create_fillup_folder: create folder in fill-up approach with SPB (which only creates images once)
    reduce_dim: reducing dimension
    visual_section: creating and reading images for others which creates images each fold
    scaler_section: scale and tranform data

3. main function to run the experiment
    run_kfold_deepmg: run experiment with k-fold
    deepmg_visual: visualize data
    run_holdout_deepmg: run experiment with holdout approach

### in order to add a new model/algorithm:
 + in models_def.py: 
        ++ add a function , e.g: knn_model, with some parameters (we call 'sp') for this function
        ++ add an else-if in "call_model" (model_def.py) with options to call to knn_model
 + in experiment.py find para_cmd(), get_default_value(), and string_to_numeric() to set 'sp' 
    if need for default value--> get_default_value, type of number-->string_to_numeric
        ++ find para_config_file() to add the 'sp' to save config when learning
        ++ find call_model() to add 'sp'
        ++ find name_log_final(): update the name for new model/algo
"""

#import utils_deepmg
#import vis_data
try:
    import vis_data
except ImportError:
    from deepmg import vis_data

try:
    import utils_deepmg as utils
except ImportError:
    from deepmg import utils_deepmg as utils

from optparse import OptionParser
import configparser

from sys import exit
import os
import numpy as np
import random as rn
import pandas as pd
import math
from time import gmtime, strftime
import time

from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score,matthews_corrcoef, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.manifold import TSNE, LocallyLinearEmbedding, SpectralEmbedding, MDS, Isomap
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, PowerTransformer
from sklearn import random_projection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier


def para_cmd ():
    '''
    READ parameters provided from the command lines with prioritizing from cmd other than config file
    '''
    parser = OptionParser()

    #experiment
    parser.add_option('-a','--type_run',help='Select a type of processing, visual (vis) transformation of the data, learning (learn) a model, testing it in another dataset, or predicting (predict) new data, or creating config file (config)')
    parser.add_option('--config_file',default='',help='Specify the path of the config file if reading parameters from files')
    parser.add_option('--seed_value_begin',help=' set the beginning seed for different runs (default:1)')
    parser.add_option('--search_already',help='if y: search existed experiments before running-->if existed, stopping the experiment (default:y)')
    parser.add_option('--cudaid',help='id of the gpu card to use when multiple exist in the server (if <=-3: use CPU, -2: use cpu if no available gpu, -1: use all gpu, >=0: id of gpu), (default:-3)')
    parser.add_option('--gpu_memory_fraction',help='gpu_memory_fraction for running by cuda (0:auto_increase based on requirement)')
    parser.add_option('--rd_pr_seed',help=' seed for random projection (default:None)')
    parser.add_option('--debug',help='show DEBUG if >0 (default:0)')
    parser.add_option('--check',help='check whether package can work properly or not (if there is any error in installation)')
    
    #grid_search
    parser.add_option('--grid_coef_time',help='choose the best coef from #coef default for tuning coeficiency (default:5)')
    parser.add_option('--cv_coef_time',help='k-cross validation for each coef for tuning coeficiency (default:4)')
    parser.add_option('--coef_ini',help='initilized coefficient for tuning coeficiency (default:255)')
    parser.add_option('--metric_selection',help='roc_auc/accuracy/neg_log_loss/grid_search_mmc for tuning coeficiency (default:roc_auc)')
    
    #io
    parser.add_option('--parent_folder_img',help='name of parent folder containing images (default:images)')
    parser.add_option('-r','--original_data_folder',help='parent folder containing data (default:data)')
    parser.add_option('-i','--data_name',help='name of dataset (default:wt2phy)')
    parser.add_option('--parent_folder_results',help='parent folder containing results (default:results)')
    parser.add_option('--save_avg_run',help='save avg performance of each run')
    parser.add_option('--save_labels',help='save labels of each fold')
    parser.add_option('--save_d',help='saving details of learning (details of each epoch --> may consume more storage')
    parser.add_option('--save_w',help='save weight mode, default:n, this might consume much storage')
    parser.add_option('--suff_fini',help='append suffix when finishing (default:ok), used to mark finished experiment')
    parser.add_option('--save_rf',help='save important features and scores for Random Forests')
    parser.add_option('--save_para',help='save parameters to config files')
    parser.add_option('--path_config_w',help='path of config to save, if empty, save to the same folder of the results')
    parser.add_option('--ext_data',help="external data with a pattern name (eg. wt2dphy), default: empty")
    parser.add_option('--save_entire_w',help='save weight of model on whole datasets (default:n)')
    parser.add_option('--save_folds',help='save results of each fold')
    parser.add_option('--sound_fini',help='play sound when finished (work on MacOS, default: n)')
    
    #Learning
    parser.add_option('-k','--n_folds',help='number of k folds (default:10), if k == 1, training on whole *_x.csv, and testing on *z_x.csv')
    parser.add_option('--test_ext',help='if==y, using external validation sets (default:n)')
    parser.add_option('--run_time',help='give the #runs (default:1)')
    #parser.add_option('--seed_ml',help='run seed for machine learning (default:n)')
    parser.add_option('--whole_run_time',help='give the #runs (default:1)')
    parser.add_option('--preprocess_img',help='support resnet50/vgg16/vgg19, none: no use    (default:none)')
    parser.add_option('--pretrained_w_path',help='path to a pretrained model, used for testing, predicting or continue to train a data')
    parser.add_option('--test_size',help='test size in holdout validation, if != 0 or != 1, then do cross validation')
    
    #model
    parser.add_option('-f','--numfilters',help='#filters/neurons for each cnn/neural layer')
    parser.add_option('-n','--numlayercnn_per_maxpool',help='#cnnlayer before each max pooling (default:1)')
    parser.add_option('--nummaxpool',help='#maxpooling_layer (default:1)')
    parser.add_option('--dropout_cnn',help=' dropout rate for CNN layer(s) (default:0)')
    parser.add_option('-d','--dropout_fc',help='dropout rate for FC layer(s) (default:0)')
    parser.add_option('--padding',help='y if use pad, others: does not use (default:n)')
    parser.add_option('--filtersize',help='the filter size (default:3)')
    parser.add_option('--poolsize',help='the pooling size (default:2)')
    parser.add_option('--model',help='model name for learning vgglike/model_lstm/resnet50/rf_model/dtc_model/gbc_model/svm_model/knn_model/none/)')
    parser.add_option('-c','--num_classes',help='#output of the network (default:1)')
    parser.add_option('-e','--epoch',help='the epoch used for training (default:500)')
    parser.add_option('--learning_rate',help=' learning rate, if -1 use default value of the optimizer (default:-1)')
    parser.add_option('--batch_size',help='batch size (default:16)')
    parser.add_option('--learning_rate_decay',help='learning rate decay (default:0)')
    parser.add_option('--momentum',help='momentum (default:0)')
    parser.add_option('-o','--optimizer',help='support sgd/adam/Adamax/RMSprop/Adagrad/Adadelta/Nadam (default:adam)')
    parser.add_option('-l','--loss_func',help='support binary_crossentropy/mae/squared_hinge/categorical_crossentropy  (default:binary_crossentropy)')
    parser.add_option('-q','--e_stop',help='#epochs with no improvement after which training will be stopped (default:5)')
    parser.add_option('--e_stop_consec',help='option to choose consective (self defined: consec) consec or  norma, default:consec')
    parser.add_option('--svm_c',help='Penalty parameter C of the error term for SVM')
    parser.add_option('--svm_kernel',help='the kernel type used in the algorithm (linear, poly, rbf, sigmoid, precomputed) (default:linear)')
    parser.add_option('--rf_n_estimators',help='The number of trees in the forest (default:500)')
    parser.add_option('--rf_max_features',help='The number of max features in the forest (default:-2: auto (sqrt), -1: all)')
    parser.add_option('--rf_max_depth',help='The number of deep tree (default:-1: None)')
    parser.add_option('--knn_n_neighbors',help='The Number of neighbors to use (default:5) in KNeighborsClassifier')
    parser.add_option('-z','--coeff',help='coeffiency (divided) for input (should use 255 for images) (default:1)')
    
    #Vis_learn
    parser.add_option('--orderf_fill',help='shuffle order of feature (not for manifolds learning): if none, use original order of data, if high: higher composition will be in the top  (default:none)')
    parser.add_option('--new_dim',help='new dimension after reduction (default:676)')
    parser.add_option('--path_vis_learn',help='read config for generating images (for qtf/mms combine fills/manifolds)')
    parser.add_option('--path_data_vis_learn',help='path to read learned data for visualizing (for qtf/mms combine fills/manifolds)')
    parser.add_option('--del0',help='if y, delete features have nothing, default: n')
    parser.add_option('--lr_visual',help='learning rate for generating visualizations (default:100.0)')
    parser.add_option('--label_visual',help="use label when using t-SNE:'': does not use")
    parser.add_option('--iter_visual',help='#iteration for run algorithm for visualization; for tsne, it should be at least 250, but  do not set so high (default:300)')
    parser.add_option('--ini_visual',help='ini for visualization algorithm (default:pca)')
    parser.add_option('--method_lle',help='method for lle embedding:  standard/ltsa/hessian/modified (default:standard)')
    parser.add_option('--eigen_solver',help='method for others (except for tsne) (default:auto)')
    parser.add_option('--cmap_vmin',help=' vmin for cmap (default:0)')
    parser.add_option('--cmap_vmax',help='vmax for cmap (default:1)')
    parser.add_option('--scale_mode',help='scaler mode for input (default: Null)')
    parser.add_option('--n_quantile',help='n_quantile in quantiletransformer (default:1000)')
    parser.add_option('--min_scale',help='minimum value for scaling (only for minmaxscaler)')
    parser.add_option('--max_scale',help='maximum value for scaling (only for minmaxscaler)')
    parser.add_option('--min_v',help='limit min for Equal Width Binning (default:0)')
    parser.add_option('--max_v',help='limit max for Equal Width Binning (default:1), affect for eqw ONLY')
    parser.add_option('--num_bin',help='the number of bins (default:10)')
    parser.add_option('--auto_v',help='if = y, auto adjust min_v and max_v (default:0), affect for eqw, and others in scaling')
    parser.add_option('--mode_pre_img',help='support caffe/tf (default:caffe)')
    
    #Visual
    parser.add_option('--channel',help='channel of images, 1: gray, 2: color (default:3)')
    parser.add_option('-m','--dim_img',help='width or height (square) of images, -1: get real size  of original images (default:-1)')
    parser.add_option('-v','--visualize_model',help='visualize the model if > 0 (default:0)')
    parser.add_option('--algo_redu',help='algorithm of dimension reduction (rd_pro/pca/fa),  if          emtpy so do not use (default:'')')
    parser.add_option('--reduc_perle',help='perlexity for tsne (default:10)')
    parser.add_option('--reduc_ini',help=' ini for reduction (default:pca) of data (only use for fillup) (default:none)')
    parser.add_option('-t','--type_emb',help='type of the embedding (default:raw): raw/bin/fill/fills/zfill/zfills/tsne')
    parser.add_option('--imp_fea',help='using important features for overlapped issues (default:none/rf/avg/max)')
    parser.add_option('-g','--label_emb',help='taxa level of labels provided in supervised embeddings kingdom=1,phylum=2,class=3,order=4,family=5, genus=6 (default:0)')
    parser.add_option('--emb_data',help='data used for embbed with dimensionality reduction: '': transformed data; o: original data')
    parser.add_option('-y','--type_bin',help='type of binnings: spb/eqw/pr/eqf (default:'')')
    parser.add_option('-p','--perlexity_neighbor',help='perlexity for tsne/#neighbors for others (default:5)')
    parser.add_option('--n_components_emb',help='ouput after embedding (default:2)')
    parser.add_option('-s','--shape_drawn',help='shape of point to illustrate data: (pixel)/ro/o(circle) (default:,)')
    parser.add_option('--fig_size',help='fig_size to contain all features, if 0:smallest which fit data,  (default:0)')
    parser.add_option('--point_size',help='point size for img (default:1)')
    parser.add_option('--colormap',help='colormaps for generating images (rainbow /nipy_spectral/jet/Paired/Reds/YlGnBu) (default:custom)')
    parser.add_option('--margin',help='margin to images (default:0)')
    parser.add_option('--alpha_v',help='1 (opaque) (default:1)')
    parser.add_option('--recreate_img',help=' if >0 rerun to create images even though they are existing (default:0)')

    (options, args) = parser.parse_args()

    #(options, args) = parser.parse_args(args=None, values=None)
    #opts_no_defaults = parser.values()
    ##print  ('opts_no_defaults'
    ##print  (opts_no_defaults
    return (options, args)

def get_default_value (options):
    '''
    set default value to para which was not set
    '''

    if not options.type_run:  #if did not set from cmd, use default value
            options.type_run = 'learn'
    if not options.config_file:  #if did not set from cmd, use default value
            options.config_file = ''
    if not options.seed_value_begin:  #if did not set from cmd, use default value
            options.seed_value_begin = '1'
    if not options.search_already:  #if did not set from cmd, use default value
            options.search_already = 'y'
    if not options.cudaid:  #if did not set from cmd, use default value
            options.cudaid = '-3'
    if not options.gpu_memory_fraction:  #if did not set from cmd, use default value
            options.gpu_memory_fraction = '0'
    if not options.rd_pr_seed:  #if did not set from cmd, use default value
            options.rd_pr_seed = 'none'
    if not options.orderf_fill:  #if did not set from cmd, use default value
            options.orderf_fill = 'none'
    if not options.debug:  #if did not set from cmd, use default value
            options.debug = 'n'
    if not options.check:  #if did not set from cmd, use default value
            options.check = 'n'
    if not options.grid_coef_time:  #if did not set from cmd, use default value
            options.grid_coef_time = '10'
    if not options.cv_coef_time:  #if did not set from cmd, use default value
            options.cv_coef_time = '4'
    if not options.coef_ini:  #if did not set from cmd, use default value
            options.coef_ini = '255'
    if not options.metric_selection:  #if did not set from cmd, use default value
            options.metric_selection = 'roc_auc'
    if not options.parent_folder_img:  #if did not set from cmd, use default value
            options.parent_folder_img = 'images'
    if not options.original_data_folder:  #if did not set from cmd, use default value
            options.original_data_folder = 'data'
    if not options.data_name:  #if did not set from cmd, use default value
            options.data_name = ''
    if not options.parent_folder_results:  #if did not set from cmd, use default value
            options.parent_folder_results = 'results'
    if not options.save_avg_run:  #if did not set from cmd, use default value
            options.save_avg_run = 'n'
    if not options.save_labels:  #if did not set from cmd, use default value
            options.save_labels = 'n'
    if not options.save_d:  #if did not set from cmd, use default value
            options.save_d = 'n'
    if not options.save_w:  #if did not set from cmd, use default value
            options.save_w = 'n'
    if not options.suff_fini:  #if did not set from cmd, use default value
            options.suff_fini = 'ok'
    if not options.save_rf:  #if did not set from cmd, use default value
            options.save_rf = 'y'
    if not options.save_para:  #if did not set from cmd, use default value
            options.save_para = 'n'
    if not options.path_config_w:  #if did not set from cmd, use default value
            options.path_config_w = ''
    if not options.ext_data:  #if did not set from cmd, use default value
            options.ext_data = ''
    if not options.save_entire_w:  #if did not set from cmd, use default value
            options.save_entire_w = 'n'
    if not options.save_folds:  #if did not set from cmd, use default value
            options.save_folds = 'y'
    if not options.sound_fini:  #if did not set from cmd, use default value
            options.sound_fini = 'n'
    if not options.n_folds:  #if did not set from cmd, use default value
            options.n_folds = '10'
    if not options.test_ext:  #if did not set from cmd, use default value
            options.test_ext = 'n'
    if not options.run_time:  #if did not set from cmd, use default value
            options.run_time = '1'
    #if not options.seed_ml:  #if did not set from cmd, use default value
    #        options.seed_ml = 'n'
    if not options.whole_run_time:  #if did not set from cmd, use default value
            options.whole_run_time = '1'
    if not options.preprocess_img:  #if did not set from cmd, use default value
            options.preprocess_img = 'none'
    if not options.pretrained_w_path:  #if did not set from cmd, use default value
            options.pretrained_w_path = ''
    if not options.test_size:  #if did not set from cmd, use default value
            options.test_size = '1'
    if not options.numfilters:  #if did not set from cmd, use default value
            options.numfilters = '64'
    if not options.numlayercnn_per_maxpool:  #if did not set from cmd, use default value
            options.numlayercnn_per_maxpool = '1'
    if not options.nummaxpool:  #if did not set from cmd, use default value
            options.nummaxpool = '1'
    if not options.dropout_cnn:  #if did not set from cmd, use default value
            options.dropout_cnn = '0'
    if not options.dropout_fc:  #if did not set from cmd, use default value
            options.dropout_fc = '0'
    if not options.padding:  #if did not set from cmd, use default value
            options.padding = 'n'
    if not options.filtersize:  #if did not set from cmd, use default value
            options.filtersize = '3'
    if not options.poolsize:  #if did not set from cmd, use default value
            options.poolsize = '2'
    if not options.model:  #if did not set from cmd, use default value
            options.model = 'fc_model'
    if not options.num_classes:  #if did not set from cmd, use default value
            options.num_classes = '1'
    if not options.epoch:  #if did not set from cmd, use default value
            options.epoch = '500'
    if not options.learning_rate:  #if did not set from cmd, use default value
            options.learning_rate = '-1'
    if not options.batch_size:  #if did not set from cmd, use default value
            options.batch_size = '16'
    if not options.learning_rate_decay:  #if did not set from cmd, use default value
            options.learning_rate_decay = '0'
    if not options.momentum:  #if did not set from cmd, use default value
            options.momentum = '0'
    if not options.optimizer:  #if did not set from cmd, use default value
            options.optimizer = 'adam'
    if not options.loss_func:  #if did not set from cmd, use default value
            options.loss_func = 'binary_crossentropy'
    if not options.e_stop:  #if did not set from cmd, use default value
            options.e_stop = '5'
    if not options.e_stop_consec:  #if did not set from cmd, use default value
            options.e_stop_consec = 'consec'
    if not options.svm_c:  #if did not set from cmd, use default value
            options.svm_c = '1'
    if not options.svm_kernel:  #if did not set from cmd, use default value
            options.svm_kernel = 'linear'
    if not options.rf_n_estimators:  #if did not set from cmd, use default value
            options.rf_n_estimators = '500'
    if not options.rf_max_features:  #if did not set from cmd, use default value
            options.rf_max_features = '-2'
    if not options.rf_max_depth:  #if did not set from cmd, use default value
            options.rf_max_depth = '-1'
    if not  options.knn_n_neighbors:
            options.knn_n_neighbors = '5'
    if not options.coeff:  #if did not set from cmd, use default value
            options.coeff = '1'
    if not options.new_dim:  #if did not set from cmd, use default value
            options.new_dim = '676'
    if not options.path_vis_learn:  #if did not set from cmd, use default value
            options.path_vis_learn = ''
    if not options.path_data_vis_learn:  #if did not set from cmd, use default value
            options.path_data_vis_learn = ''
    if not options.del0:  #if did not set from cmd, use default value
            options.del0 = 'n'
    if not options.lr_visual:  #if did not set from cmd, use default value
            options.lr_visual = '100'
    if not options.label_visual:  #if did not set from cmd, use default value
            options.label_visual = ''
    if not options.iter_visual:  #if did not set from cmd, use default value
            options.iter_visual = '300'
    if not options.ini_visual:  #if did not set from cmd, use default value
            options.ini_visual = 'pca'
    if not options.method_lle:  #if did not set from cmd, use default value
            options.method_lle = 'standard'
    if not options.eigen_solver:  #if did not set from cmd, use default value
            options.eigen_solver = 'auto'
    if not options.cmap_vmin:  #if did not set from cmd, use default value
            options.cmap_vmin = '0'
    if not options.cmap_vmax:  #if did not set from cmd, use default value
            options.cmap_vmax = '1'
    if not options.scale_mode:  #if did not set from cmd, use default value
            options.scale_mode = ''
    if not options.n_quantile:  #if did not set from cmd, use default value
            options.n_quantile = '1000'
    if not options.min_scale:  #if did not set from cmd, use default value
            options.min_scale = '0'
    if not options.max_scale:  #if did not set from cmd, use default value
            options.max_scale = '1'
    if not options.min_v:  #if did not set from cmd, use default value
            options.min_v = '0'
    if not options.max_v:  #if did not set from cmd, use default value
            options.max_v = '1'
    if not options.num_bin:  #if did not set from cmd, use default value
            options.num_bin = '10'
    if not options.auto_v:  #if did not set from cmd, use default value
            options.auto_v = 'n'
    if not options.mode_pre_img:  #if did not set from cmd, use default value
            options.mode_pre_img = 'caffe'
    if not options.channel:  #if did not set from cmd, use default value
            options.channel = '3'
    if not options.dim_img:  #if did not set from cmd, use default value
            options.dim_img = '-1'
    if not options.visualize_model:  #if did not set from cmd, use default value
            options.visualize_model = 'n'
    if not options.algo_redu:  #if did not set from cmd, use default value
            options.algo_redu = 'none'
    if not options.reduc_perle:  #if did not set from cmd, use default value
            options.reduc_perle = '10'
    if not options.reduc_ini:  #if did not set from cmd, use default value
            options.reduc_ini = 'pca'
    if not options.type_emb:  #if did not set from cmd, use default value
            options.type_emb = 'raw'
    if not options.imp_fea:  #if did not set from cmd, use default value
            options.imp_fea = 'none'
    if not options.label_emb:  #if did not set from cmd, use default value
            options.label_emb = '0'
    if not options.emb_data:  #if did not set from cmd, use default value
            options.emb_data = ''
    if not options.type_bin:  #if did not set from cmd, use default value
            options.type_bin = ''
    if not options.perlexity_neighbor:  #if did not set from cmd, use default value
            options.perlexity_neighbor = '5'
    if not options.n_components_emb:  #if did not set from cmd, use default value
            options.n_components_emb = '2'
    if not options.shape_drawn:  #if did not set from cmd, use default value
            options.shape_drawn = ','
    if not options.fig_size:  #if did not set from cmd, use default value
            options.fig_size = '0'
    if not options.point_size:  #if did not set from cmd, use default value
            options.point_size = '1'
    if not options.colormap:  #if did not set from cmd, use default value
            options.colormap = ''
    if not options.margin:  #if did not set from cmd, use default value
            options.margin = '0'
    if not options.alpha_v:  #if did not set from cmd, use default value
            options.alpha_v = '1'
    if not options.recreate_img:  #if did not set from cmd, use default value
            options.recreate_img = '0'

    return options

def para_config_file (options):
    '''
    READ parameters provided from a configuration file
    these parameters in the configuration file are fixed, 
    modify these parameters from cmd will not work, only can modify in the configuration file

    Args:
        options (array): array of parameters
    '''

    config = configparser.configparser()
    #read config file, to retrieve the value then use config.get
    config.read (options.config_file)

    if not options.type_run:  #if did not set from cmd
        if config.has_option('experiment','type_run'): #if avaiable from config file
            options.type_run = config.get('experiment','type_run')
        else: #other cases, use default value
            options.type_run = 'learn'
    if not options.config_file:  #if did not set from cmd
        if config.has_option('experiment','config_file'): #if avaiable from config file
            options.config_file = config.get('experiment','config_file')
        else: #other cases, use default value
            options.config_file = ''
    if not options.seed_value_begin:  #if did not set from cmd
        if config.has_option('experiment','seed_value_begin'): #if avaiable from config file
            options.seed_value_begin = config.get('experiment','seed_value_begin')
        else: #other cases, use default value
            options.seed_value_begin = '1'
    if not options.search_already:  #if did not set from cmd
        if config.has_option('experiment','search_already'): #if avaiable from config file
            options.search_already = config.get('experiment','search_already')
        else: #other cases, use default value
            options.search_already = 'y'
    if not options.cudaid:  #if did not set from cmd
        if config.has_option('experiment','cudaid'): #if avaiable from config file
            options.cudaid = config.get('experiment','cudaid')
        else: #other cases, use default value
            options.cudaid = '-3'
    if not options.gpu_memory_fraction:  #if did not set from cmd
        if config.has_option('experiment','gpu_memory_fraction'): #if avaiable from config file
            options.gpu_memory_fraction = config.get('experiment','gpu_memory_fraction')
        else: #other cases, use default value
            options.gpu_memory_fraction = '0'
    if not options.rd_pr_seed:  #if did not set from cmd
        if config.has_option('experiment','rd_pr_seed'): #if avaiable from config file
            options.rd_pr_seed = config.get('experiment','rd_pr_seed')
        else: #other cases, use default value
            options.rd_pr_seed = 'none'
    if not options.orderf_fill:  #if did not set from cmd
        if config.has_option('experiment','orderf_fill'): #if avaiable from config file
            options.orderf_fill = config.get('experiment','orderf_fill')
        else: #other cases, use default value
            options.orderf_fill = 'none'
    if not options.debug:  #if did not set from cmd
        if config.has_option('experiment','debug'): #if avaiable from config file
            options.debug = config.get('experiment','debug')
        else: #other cases, use default value
            options.debug = 'n'
    if not options.check:  #if did not set from cmd
        if config.has_option('experiment','check'): #if avaiable from config file
            options.check = config.get('experiment','check')
        else: #other cases, use default value
            options.check = 'n'
    if not options.grid_coef_time:  #if did not set from cmd
        if config.has_option('grid_search','grid_coef_time'): #if avaiable from config file
            options.grid_coef_time = config.get('grid_search','grid_coef_time')
        else: #other cases, use default value
            options.grid_coef_time = '10'
    if not options.cv_coef_time:  #if did not set from cmd
        if config.has_option('grid_search','cv_coef_time'): #if avaiable from config file
            options.cv_coef_time = config.get('grid_search','cv_coef_time')
        else: #other cases, use default value
            options.cv_coef_time = '4'
    if not options.coef_ini:  #if did not set from cmd
        if config.has_option('grid_search','coef_ini'): #if avaiable from config file
            options.coef_ini = config.get('grid_search','coef_ini')
        else: #other cases, use default value
            options.coef_ini = '255'
    if not options.metric_selection:  #if did not set from cmd
        if config.has_option('grid_search','metric_selection'): #if avaiable from config file
            options.metric_selection = config.get('grid_search','metric_selection')
        else: #other cases, use default value
            options.metric_selection = 'roc_auc'
    if not options.parent_folder_img:  #if did not set from cmd
        if config.has_option('io','parent_folder_img'): #if avaiable from config file
            options.parent_folder_img = config.get('io','parent_folder_img')
        else: #other cases, use default value
            options.parent_folder_img = 'images'
    if not options.original_data_folder:  #if did not set from cmd
        if config.has_option('io','original_data_folder'): #if avaiable from config file
            options.original_data_folder = config.get('io','original_data_folder')
        else: #other cases, use default value
            options.original_data_folder = 'data'
    if not options.data_name:  #if did not set from cmd
        if config.has_option('io','data_name'): #if avaiable from config file
            options.data_name = config.get('io','data_name')
        else: #other cases, use default value
            options.data_name = ''
    if not options.parent_folder_results:  #if did not set from cmd
        if config.has_option('io','parent_folder_results'): #if avaiable from config file
            options.parent_folder_results = config.get('io','parent_folder_results')
        else: #other cases, use default value
            options.parent_folder_results = 'results'
    if not options.save_avg_run:  #if did not set from cmd
        if config.has_option('io','save_avg_run'): #if avaiable from config file
            options.save_avg_run = config.get('io','save_avg_run')
        else: #other cases, use default value
            options.save_avg_run = 'n'
    if not options.save_labels:  #if did not set from cmd
        if config.has_option('io','save_labels'): #if avaiable from config file
            options.save_labels = config.get('io','save_labels')
        else: #other cases, use default value
            options.save_labels = 'n'
    if not options.save_d:  #if did not set from cmd
        if config.has_option('io','save_d'): #if avaiable from config file
            options.save_d = config.get('io','save_d')
        else: #other cases, use default value
            options.save_d = 'n'
    if not options.save_w:  #if did not set from cmd
        if config.has_option('io','save_w'): #if avaiable from config file
            options.save_w = config.get('io','save_w')
        else: #other cases, use default value
            options.save_w = 'n'
    if not options.suff_fini:  #if did not set from cmd
        if config.has_option('io','suff_fini'): #if avaiable from config file
            options.suff_fini = config.get('io','suff_fini')
        else: #other cases, use default value
            options.suff_fini = 'ok'
    if not options.save_rf:  #if did not set from cmd
        if config.has_option('io','save_rf'): #if avaiable from config file
            options.save_rf = config.get('io','save_rf')
        else: #other cases, use default value
            options.save_rf = 'y'
    if not options.save_para:  #if did not set from cmd
        if config.has_option('io','save_para'): #if avaiable from config file
            options.save_para = config.get('io','save_para')
        else: #other cases, use default value
            options.save_para = 'n'
    if not options.path_config_w:  #if did not set from cmd
        if config.has_option('io','path_config_w'): #if avaiable from config file
            options.path_config_w = config.get('io','path_config_w')
        else: #other cases, use default value
            options.path_config_w = ''
    if not options.ext_data:  #if did not set from cmd
        if config.has_option('io','ext_data'): #if avaiable from config file
            options.ext_data = config.get('io','ext_data')
        else: #other cases, use default value
            options.ext_data = ''
    if not options.save_entire_w:  #if did not set from cmd
        if config.has_option('io','save_entire_w'): #if avaiable from config file
            options.save_entire_w = config.get('io','save_entire_w')
        else: #other cases, use default value
            options.save_entire_w = 'n'
    if not options.save_folds:  #if did not set from cmd
        if config.has_option('io','save_folds'): #if avaiable from config file
            options.save_folds = config.get('io','save_folds')
        else: #other cases, use default value
            options.save_folds = 'y'
    if not options.sound_fini:  #if did not set from cmd
        if config.has_option('io','sound_fini'): #if avaiable from config file
            options.sound_fini = config.get('io','sound_fini')
        else: #other cases, use default value
            options.sound_fini = 'n'
    if not options.n_folds:  #if did not set from cmd
        if config.has_option('Learning','n_folds'): #if avaiable from config file
            options.n_folds = config.get('Learning','n_folds')
        else: #other cases, use default value
            options.n_folds = '10'
    if not options.test_ext:  #if did not set from cmd
        if config.has_option('Learning','test_ext'): #if avaiable from config file
            options.test_ext = config.get('Learning','test_ext')
        else: #other cases, use default value
            options.test_ext = 'n'
    if not options.run_time:  #if did not set from cmd
        if config.has_option('Learning','run_time'): #if avaiable from config file
            options.run_time = config.get('Learning','run_time')
        else: #other cases, use default value
            options.run_time = '1'
    #if not options.seed_ml:  #if did not set from cmd
    #    if config.has_option('Learning','seed_ml'): #if avaiable from config file
    #        options.seed_ml = config.get('Learning','seed_ml')
    #    else: #other cases, use default value
    #        options.seed_ml = 'n'
    if not options.whole_run_time:  #if did not set from cmd
        if config.has_option('Learning','whole_run_time'): #if avaiable from config file
            options.whole_run_time = config.get('Learning','whole_run_time')
        else: #other cases, use default value
            options.whole_run_time = '1'
    if not options.preprocess_img:  #if did not set from cmd
        if config.has_option('Learning','preprocess_img'): #if avaiable from config file
            options.preprocess_img = config.get('Learning','preprocess_img')
        else: #other cases, use default value
            options.preprocess_img = 'none'
    if not options.pretrained_w_path:  #if did not set from cmd
        if config.has_option('Learning','pretrained_w_path'): #if avaiable from config file
            options.pretrained_w_path = config.get('Learning','pretrained_w_path')
        else: #other cases, use default value
            options.pretrained_w_path = ''
    if not options.test_size:  #if did not set from cmd
        if config.has_option('Learning','test_size'): #if avaiable from config file
            options.test_size = config.get('Learning','test_size')
        else: #other cases, use default value
            options.test_size = '1'
    if not options.numfilters:  #if did not set from cmd
        if config.has_option('model','numfilters'): #if avaiable from config file
            options.numfilters = config.get('model','numfilters')
        else: #other cases, use default value
            options.numfilters = '64'
    if not options.numlayercnn_per_maxpool:  #if did not set from cmd
        if config.has_option('model','numlayercnn_per_maxpool'): #if avaiable from config file
            options.numlayercnn_per_maxpool = config.get('model','numlayercnn_per_maxpool')
        else: #other cases, use default value
            options.numlayercnn_per_maxpool = '1'
    if not options.nummaxpool:  #if did not set from cmd
        if config.has_option('model','nummaxpool'): #if avaiable from config file
            options.nummaxpool = config.get('model','nummaxpool')
        else: #other cases, use default value
            options.nummaxpool = '1'
    if not options.dropout_cnn:  #if did not set from cmd
        if config.has_option('model','dropout_cnn'): #if avaiable from config file
            options.dropout_cnn = config.get('model','dropout_cnn')
        else: #other cases, use default value
            options.dropout_cnn = '0'
    if not options.dropout_fc:  #if did not set from cmd
        if config.has_option('model','dropout_fc'): #if avaiable from config file
            options.dropout_fc = config.get('model','dropout_fc')
        else: #other cases, use default value
            options.dropout_fc = '0'
    if not options.padding:  #if did not set from cmd
        if config.has_option('model','padding'): #if avaiable from config file
            options.padding = config.get('model','padding')
        else: #other cases, use default value
            options.padding = 'n'
    if not options.filtersize:  #if did not set from cmd
        if config.has_option('model','filtersize'): #if avaiable from config file
            options.filtersize = config.get('model','filtersize')
        else: #other cases, use default value
            options.filtersize = '3'
    if not options.poolsize:  #if did not set from cmd
        if config.has_option('model','poolsize'): #if avaiable from config file
            options.poolsize = config.get('model','poolsize')
        else: #other cases, use default value
            options.poolsize = '2'
    if not options.model:  #if did not set from cmd
        if config.has_option('model','model'): #if avaiable from config file
            options.model = config.get('model','model')
        else: #other cases, use default value
            options.model = 'fc_model'
    if not options.num_classes:  #if did not set from cmd
        if config.has_option('model','num_classes'): #if avaiable from config file
            options.num_classes = config.get('model','num_classes')
        else: #other cases, use default value
            options.num_classes = '1'
    if not options.epoch:  #if did not set from cmd
        if config.has_option('model','epoch'): #if avaiable from config file
            options.epoch = config.get('model','epoch')
        else: #other cases, use default value
            options.epoch = '500'
    if not options.learning_rate:  #if did not set from cmd
        if config.has_option('model','learning_rate'): #if avaiable from config file
            options.learning_rate = config.get('model','learning_rate')
        else: #other cases, use default value
            options.learning_rate = '-1'
    if not options.batch_size:  #if did not set from cmd
        if config.has_option('model','batch_size'): #if avaiable from config file
            options.batch_size = config.get('model','batch_size')
        else: #other cases, use default value
            options.batch_size = '16'
    if not options.learning_rate_decay:  #if did not set from cmd
        if config.has_option('model','learning_rate_decay'): #if avaiable from config file
            options.learning_rate_decay = config.get('model','learning_rate_decay')
        else: #other cases, use default value
            options.learning_rate_decay = '0'
    if not options.momentum:  #if did not set from cmd
        if config.has_option('model','momentum'): #if avaiable from config file
            options.momentum = config.get('model','momentum')
        else: #other cases, use default value
            options.momentum = '0'
    if not options.optimizer:  #if did not set from cmd
        if config.has_option('model','optimizer'): #if avaiable from config file
            options.optimizer = config.get('model','optimizer')
        else: #other cases, use default value
            options.optimizer = 'adam'
    if not options.loss_func:  #if did not set from cmd
        if config.has_option('model','loss_func'): #if avaiable from config file
            options.loss_func = config.get('model','loss_func')
        else: #other cases, use default value
            options.loss_func = 'binary_crossentropy'
    if not options.e_stop:  #if did not set from cmd
        if config.has_option('model','e_stop'): #if avaiable from config file
            options.e_stop = config.get('model','e_stop')
        else: #other cases, use default value
            options.e_stop = '5'
    if not options.e_stop_consec:  #if did not set from cmd
        if config.has_option('model','e_stop_consec'): #if avaiable from config file
            options.e_stop_consec = config.get('model','e_stop_consec')
        else: #other cases, use default value
            options.e_stop_consec = 'consec'
    if not options.svm_c:  #if did not set from cmd
        if config.has_option('model','svm_c'): #if avaiable from config file
            options.svm_c = config.get('model','svm_c')
        else: #other cases, use default value
            options.svm_c = '1'
    if not options.svm_kernel:  #if did not set from cmd
        if config.has_option('model','svm_kernel'): #if avaiable from config file
            options.svm_kernel = config.get('model','svm_kernel')
        else: #other cases, use default value
            options.svm_kernel = 'linear'
    if not options.rf_n_estimators:  #if did not set from cmd
        if config.has_option('model','rf_n_estimators'): #if avaiable from config file
            options.rf_n_estimators = config.get('model','rf_n_estimators')
        else: #other cases, use default value
            options.rf_n_estimators = '500'
    if not options.rf_max_features:  #if did not set from cmd
        if config.has_option('model','rf_max_features'): #if avaiable from config file
            options.rf_max_features = config.get('model','rf_max_features')
        else: #other cases, use default value
            options.rf_max_features = '-2'
    if not options.rf_max_depth:  #if did not set from cmd
        if config.has_option('model','rf_max_depth'): #if avaiable from config file
            options.rf_max_depth = config.get('model','rf_max_depth')
        else: #other cases, use default value
            options.rf_max_depth = '-1'
    if not options.knn_n_neighbors:  #if did not set from cmd
        if config.has_option('model','knn_n_neighbors'): #if avaiable from config file
            options.knn_n_neighbors = config.get('model','knn_n_neighbors')
        else: #other cases, use default value
            options.knn_n_neighbors = '5'
    if not options.coeff:  #if did not set from cmd
        if config.has_option('model','coeff'): #if avaiable from config file
            options.coeff = config.get('model','coeff')
        else: #other cases, use default value
            options.coeff = '1'
    if not options.new_dim:  #if did not set from cmd
        if config.has_option('Vis_learn','new_dim'): #if avaiable from config file
            options.new_dim = config.get('Vis_learn','new_dim')
        else: #other cases, use default value
            options.new_dim = '676'
    if not options.path_vis_learn:  #if did not set from cmd
        if config.has_option('Vis_learn','path_vis_learn'): #if avaiable from config file
            options.path_vis_learn = config.get('Vis_learn','path_vis_learn')
        else: #other cases, use default value
            options.path_vis_learn = ''
    if not options.path_data_vis_learn:  #if did not set from cmd
        if config.has_option('Vis_learn','path_data_vis_learn'): #if avaiable from config file
            options.path_data_vis_learn = config.get('Vis_learn','path_data_vis_learn')
        else: #other cases, use default value
            options.path_data_vis_learn = ''
    if not options.del0:  #if did not set from cmd
        if config.has_option('Vis_learn','del0'): #if avaiable from config file
            options.del0 = config.get('Vis_learn','del0')
        else: #other cases, use default value
            options.del0 = 'n'
    if not options.lr_visual:  #if did not set from cmd
        if config.has_option('Vis_learn','lr_visual'): #if avaiable from config file
            options.lr_visual = config.get('Vis_learn','lr_visual')
        else: #other cases, use default value
            options.lr_visual = '100'
    if not options.label_visual:  #if did not set from cmd
        if config.has_option('Vis_learn','label_visual'): #if avaiable from config file
            options.label_visual = config.get('Vis_learn','label_visual')
        else: #other cases, use default value
            options.label_visual = ''
    if not options.iter_visual:  #if did not set from cmd
        if config.has_option('Vis_learn','iter_visual'): #if avaiable from config file
            options.iter_visual = config.get('Vis_learn','iter_visual')
        else: #other cases, use default value
            options.iter_visual = '300'
    if not options.ini_visual:  #if did not set from cmd
        if config.has_option('Vis_learn','ini_visual'): #if avaiable from config file
            options.ini_visual = config.get('Vis_learn','ini_visual')
        else: #other cases, use default value
            options.ini_visual = 'pca'
    if not options.method_lle:  #if did not set from cmd
        if config.has_option('Vis_learn','method_lle'): #if avaiable from config file
            options.method_lle = config.get('Vis_learn','method_lle')
        else: #other cases, use default value
            options.method_lle = 'standard'
    if not options.eigen_solver:  #if did not set from cmd
        if config.has_option('Vis_learn','eigen_solver'): #if avaiable from config file
            options.eigen_solver = config.get('Vis_learn','eigen_solver')
        else: #other cases, use default value
            options.eigen_solver = 'auto'
    if not options.cmap_vmin:  #if did not set from cmd
        if config.has_option('Vis_learn','cmap_vmin'): #if avaiable from config file
            options.cmap_vmin = config.get('Vis_learn','cmap_vmin')
        else: #other cases, use default value
            options.cmap_vmin = '0'
    if not options.cmap_vmax:  #if did not set from cmd
        if config.has_option('Vis_learn','cmap_vmax'): #if avaiable from config file
            options.cmap_vmax = config.get('Vis_learn','cmap_vmax')
        else: #other cases, use default value
            options.cmap_vmax = '1'
    if not options.scale_mode:  #if did not set from cmd
        if config.has_option('Vis_learn','scale_mode'): #if avaiable from config file
            options.scale_mode = config.get('Vis_learn','scale_mode')
        else: #other cases, use default value
            options.scale_mode = ''
    if not options.n_quantile:  #if did not set from cmd
        if config.has_option('Vis_learn','n_quantile'): #if avaiable from config file
            options.n_quantile = config.get('Vis_learn','n_quantile')
        else: #other cases, use default value
            options.n_quantile = '1000'
    if not options.min_scale:  #if did not set from cmd
        if config.has_option('Vis_learn','min_scale'): #if avaiable from config file
            options.min_scale = config.get('Vis_learn','min_scale')
        else: #other cases, use default value
            options.min_scale = '0'
    if not options.max_scale:  #if did not set from cmd
        if config.has_option('Vis_learn','max_scale'): #if avaiable from config file
            options.max_scale = config.get('Vis_learn','max_scale')
        else: #other cases, use default value
            options.max_scale = '1'
    if not options.min_v:  #if did not set from cmd
        if config.has_option('Vis_learn','min_v'): #if avaiable from config file
            options.min_v = config.get('Vis_learn','min_v')
        else: #other cases, use default value
            options.min_v = '0'
    if not options.max_v:  #if did not set from cmd
        if config.has_option('Vis_learn','max_v'): #if avaiable from config file
            options.max_v = config.get('Vis_learn','max_v')
        else: #other cases, use default value
            options.max_v = '1'
    if not options.num_bin:  #if did not set from cmd
        if config.has_option('Vis_learn','num_bin'): #if avaiable from config file
            options.num_bin = config.get('Vis_learn','num_bin')
        else: #other cases, use default value
            options.num_bin = '10'
    if not options.auto_v:  #if did not set from cmd
        if config.has_option('Vis_learn','auto_v'): #if avaiable from config file
            options.auto_v = config.get('Vis_learn','auto_v')
        else: #other cases, use default value
            options.auto_v = 'n'
    if not options.mode_pre_img:  #if did not set from cmd
        if config.has_option('Vis_learn','mode_pre_img'): #if avaiable from config file
            options.mode_pre_img = config.get('Vis_learn','mode_pre_img')
        else: #other cases, use default value
            options.mode_pre_img = 'caffe'
    if not options.channel:  #if did not set from cmd
        if config.has_option('Visual','channel'): #if avaiable from config file
            options.channel = config.get('Visual','channel')
        else: #other cases, use default value
            options.channel = '3'
    if not options.dim_img:  #if did not set from cmd
        if config.has_option('Visual','dim_img'): #if avaiable from config file
            options.dim_img = config.get('Visual','dim_img')
        else: #other cases, use default value
            options.dim_img = '-1'
    if not options.visualize_model:  #if did not set from cmd
        if config.has_option('Visual','visualize_model'): #if avaiable from config file
            options.visualize_model = config.get('Visual','visualize_model')
        else: #other cases, use default value
            options.visualize_model = 'n'
    if not options.algo_redu:  #if did not set from cmd
        if config.has_option('Visual','algo_redu'): #if avaiable from config file
            options.algo_redu = config.get('Visual','algo_redu')
        else: #other cases, use default value
            options.algo_redu = 'none'
    if not options.reduc_perle:  #if did not set from cmd
        if config.has_option('Visual','reduc_perle'): #if avaiable from config file
            options.reduc_perle = config.get('Visual','reduc_perle')
        else: #other cases, use default value
            options.reduc_perle = '10'
    if not options.reduc_ini:  #if did not set from cmd
        if config.has_option('Visual','reduc_ini'): #if avaiable from config file
            options.reduc_ini = config.get('Visual','reduc_ini')
        else: #other cases, use default value
            options.reduc_ini = 'pca'
    if not options.type_emb:  #if did not set from cmd
        if config.has_option('Visual','type_emb'): #if avaiable from config file
            options.type_emb = config.get('Visual','type_emb')
        else: #other cases, use default value
            options.type_emb = 'raw'
    if not options.imp_fea:  #if did not set from cmd
        if config.has_option('Visual','imp_fea'): #if avaiable from config file
            options.imp_fea = config.get('Visual','imp_fea')
        else: #other cases, use default value
            options.imp_fea = 'none'
    if not options.label_emb:  #if did not set from cmd
        if config.has_option('Visual','label_emb'): #if avaiable from config file
            options.label_emb = config.get('Visual','label_emb')
        else: #other cases, use default value
            options.label_emb = '0'
    if not options.emb_data:  #if did not set from cmd
        if config.has_option('Visual','emb_data'): #if avaiable from config file
            options.emb_data = config.get('Visual','emb_data')
        else: #other cases, use default value
            options.emb_data = ''
    if not options.type_bin:  #if did not set from cmd
        if config.has_option('Visual','type_bin'): #if avaiable from config file
            options.type_bin = config.get('Visual','type_bin')
        else: #other cases, use default value
            options.type_bin = ''
    if not options.perlexity_neighbor:  #if did not set from cmd
        if config.has_option('Visual','perlexity_neighbor'): #if avaiable from config file
            options.perlexity_neighbor = config.get('Visual','perlexity_neighbor')
        else: #other cases, use default value
            options.perlexity_neighbor = '5'
    if not options.n_components_emb:  #if did not set from cmd
        if config.has_option('Visual','n_components_emb'): #if avaiable from config file
            options.n_components_emb = config.get('Visual','n_components_emb')
        else: #other cases, use default value
            options.n_components_emb = '2'
    if not options.shape_drawn:  #if did not set from cmd
        if config.has_option('Visual','shape_drawn'): #if avaiable from config file
            options.shape_drawn = config.get('Visual','shape_drawn')
        else: #other cases, use default value
            options.shape_drawn = ','
    if not options.fig_size:  #if did not set from cmd
        if config.has_option('Visual','fig_size'): #if avaiable from config file
            options.fig_size = config.get('Visual','fig_size')
        else: #other cases, use default value
            options.fig_size = '0'
    if not options.point_size:  #if did not set from cmd
        if config.has_option('Visual','point_size'): #if avaiable from config file
            options.point_size = config.get('Visual','point_size')
        else: #other cases, use default value
            options.point_size = '1'
    if not options.colormap:  #if did not set from cmd
        if config.has_option('Visual','colormap'): #if avaiable from config file
            options.colormap = config.get('Visual','colormap')
        else: #other cases, use default value
            options.colormap = ''
    if not options.margin:  #if did not set from cmd
        if config.has_option('Visual','margin'): #if avaiable from config file
            options.margin = config.get('Visual','margin')
        else: #other cases, use default value
            options.margin = '0'
    if not options.alpha_v:  #if did not set from cmd
        if config.has_option('Visual','alpha_v'): #if avaiable from config file
            options.alpha_v = config.get('Visual','alpha_v')
        else: #other cases, use default value
            options.alpha_v = '1'
    if not options.recreate_img:  #if did not set from cmd
        if config.has_option('Visual','recreate_img'): #if avaiable from config file
            options.recreate_img = config.get('Visual','recreate_img')
        else: #other cases, use default value
            options.recreate_img = '0'
 
def string_to_numeric(options):
    
    '''
    convert options which own numberic type
    '''

    options.seed_value_begin =  int(options.seed_value_begin)

    options.cudaid =  int(options.cudaid)
    options.gpu_memory_fraction =  float(options.gpu_memory_fraction)




    options.grid_coef_time =  int(options.grid_coef_time)
    options.cv_coef_time =  int(options.cv_coef_time)
    options.coef_ini =  float(options.coef_ini)














    options.n_folds =  int(options.n_folds)

    options.run_time =  int(options.run_time)
    
    options.whole_run_time =  int(options.whole_run_time)


    options.test_size =  float(options.test_size)
    options.numfilters =  int(options.numfilters)
    options.numlayercnn_per_maxpool =  int(options.numlayercnn_per_maxpool)
    options.nummaxpool =  int(options.nummaxpool)
    options.dropout_cnn =  float(options.dropout_cnn)
    options.dropout_fc =  float(options.dropout_fc)

    options.filtersize =  int(options.filtersize)
    options.poolsize =  int(options.poolsize)

    options.num_classes =  int(options.num_classes)
    options.epoch =  int(options.epoch)
    options.learning_rate =  float(options.learning_rate)
    options.batch_size =  int(options.batch_size)
    options.learning_rate_decay =  float(options.learning_rate_decay)
    options.momentum =  float(options.momentum)


    options.e_stop =  int(options.e_stop)

    options.svm_c =  float(options.svm_c)

    options.rf_n_estimators =  int(options.rf_n_estimators)
    options.rf_max_features =  int(options.rf_max_features)

    options.rf_max_depth =  int(options.rf_max_depth)
    options.knn_n_neighbors =  int(options.knn_n_neighbors)
    options.coeff =  float(options.coeff)
    options.new_dim =  int(options.new_dim)



    options.lr_visual =  float(options.lr_visual)

    options.iter_visual =  int(options.iter_visual)



    options.cmap_vmin =  float(options.cmap_vmin)
    options.cmap_vmax =  float(options.cmap_vmax)

    options.n_quantile =  int(options.n_quantile)
    options.min_scale =  float(options.min_scale)
    options.max_scale =  float(options.max_scale)
    options.min_v =  float(options.min_v)
    options.max_v =  float(options.max_v)
    options.num_bin =  int(options.num_bin)


    options.channel =  int(options.channel)
    options.dim_img =  int(options.dim_img)


    options.reduc_perle =  int(options.reduc_perle)



    options.label_emb =  int(options.label_emb)


    options.perlexity_neighbor =  int(options.perlexity_neighbor)
    options.n_components_emb =  int(options.n_components_emb)

    options.fig_size =  int(options.fig_size)
    options.point_size =  float(options.point_size)


    options.margin =  float(options.margin)
    options.alpha_v =  float(options.alpha_v)
    options.recreate_img =  int(options.recreate_img)   

    return options

def write_para(options,configfile):
    '''
    WRITE parameters TO FILE configfile (included path)
    
    Args:
        options (array): array of parameters
        configfile (string): file to write
    '''
    #Config = configparser.configparser()
    Config = configparser.ConfigParser()    

    #create section, each section might contain one or more elements, eg. experiment includes type_run, config_file....
    Config.add_section('experiment')
    Config.add_section('grid_search')
    Config.add_section('Learning')
    Config.add_section('io')    
    Config.add_section('Vis_learn')
    Config.add_section('model')
    Config.add_section('Visual')

    Config.set('experiment','type_run',options.type_run)
    Config.set('experiment','config_file',options.config_file)
    Config.set('experiment','seed_value_begin',str(options.seed_value_begin))
    Config.set('experiment','search_already',options.search_already)
    Config.set('experiment','cudaid',str(options.cudaid))
    Config.set('experiment','gpu_memory_fraction',str(options.gpu_memory_fraction))
    Config.set('experiment','rd_pr_seed',str(options.rd_pr_seed))
    Config.set('experiment','orderf_fill',str(options.orderf_fill))
    Config.set('experiment','debug',options.debug)
    Config.set('experiment','check',options.check)
    Config.set('grid_search','grid_coef_time',str(options.grid_coef_time))
    Config.set('grid_search','cv_coef_time',str(options.cv_coef_time))
    Config.set('grid_search','coef_ini',str(options.coef_ini))
    Config.set('grid_search','metric_selection',options.metric_selection)
    Config.set('io','parent_folder_img',options.parent_folder_img)
    Config.set('io','original_data_folder',options.original_data_folder)
    Config.set('io','data_name',options.data_name)
    Config.set('io','parent_folder_results',options.parent_folder_results)
    Config.set('io','save_avg_run',options.save_avg_run)
    Config.set('io','save_labels',options.save_labels)
    Config.set('io','save_d',options.save_d)
    Config.set('io','save_w',options.save_w)
    Config.set('io','suff_fini',options.suff_fini)
    Config.set('io','save_rf',options.save_rf)
    Config.set('io','save_para',options.save_para)
    Config.set('io','path_config_w',options.path_config_w)
    Config.set('io','ext_data',options.path_config_w)
    Config.set('io','save_entire_w',options.save_entire_w)
    Config.set('io','save_folds',options.save_folds)
    Config.set('io','sound_fini',options.sound_fini)
    Config.set('Learning','n_folds',str(options.n_folds))
    Config.set('Learning','test_ext',options.test_ext)
    Config.set('Learning','run_time',str(options.run_time))
    Config.set('Learning','whole_run_time',str(options.whole_run_time))
    Config.set('Learning','preprocess_img',options.preprocess_img)
    Config.set('Learning','pretrained_w_path',options.pretrained_w_path)
    Config.set('Learning','test_size',str(options.test_size))
    Config.set('model','numfilters',str(options.numfilters))
    Config.set('model','numlayercnn_per_maxpool',str(options.numlayercnn_per_maxpool))
    Config.set('model','nummaxpool',str(options.nummaxpool))
    Config.set('model','dropout_cnn',str(options.dropout_cnn))
    Config.set('model','dropout_fc',str(options.dropout_fc))
    Config.set('model','padding',options.padding)
    Config.set('model','filtersize',str(options.filtersize))
    Config.set('model','poolsize',str(options.poolsize))
    Config.set('model','model',options.model)
    Config.set('model','num_classes',str(options.num_classes))
    Config.set('model','epoch',str(options.epoch))
    Config.set('model','learning_rate',str(options.learning_rate))
    Config.set('model','batch_size',str(options.batch_size))
    Config.set('model','learning_rate_decay',str(options.learning_rate_decay))
    Config.set('model','momentum',str(options.momentum))
    Config.set('model','optimizer',options.optimizer)
    Config.set('model','loss_func',options.loss_func)
    Config.set('model','e_stop',str(options.e_stop))
    Config.set('model','e_stop_consec',str(options.e_stop_consec))
    Config.set('model','svm_c',str(options.svm_c))
    Config.set('model','svm_kernel',options.svm_kernel)
    Config.set('model','rf_n_estimators',str(options.rf_n_estimators))
    Config.set('model','rf_max_features',str(options.rf_max_features))
    Config.set('model','rf_max_depth',str(options.rf_max_depth))
    Config.set('model','coeff',str(options.coeff))
    Config.set('Vis_learn','new_dim',str(options.new_dim))
    Config.set('Vis_learn','path_vis_learn',options.path_vis_learn)
    Config.set('Vis_learn','path_data_vis_learn',options.path_data_vis_learn)
    Config.set('Vis_learn','del0',options.del0)
    Config.set('Vis_learn','lr_visual',str(options.lr_visual))
    Config.set('Vis_learn','label_visual',options.label_visual)
    Config.set('Vis_learn','iter_visual',str(options.iter_visual))
    Config.set('Vis_learn','ini_visual',str(options.ini_visual))
    Config.set('Vis_learn','method_lle',options.method_lle)
    Config.set('Vis_learn','eigen_solver',options.eigen_solver)
    Config.set('Vis_learn','cmap_vmin',str(options.cmap_vmin))
    Config.set('Vis_learn','cmap_vmax',str(options.cmap_vmax))
    Config.set('Vis_learn','scale_mode',options.scale_mode)
    Config.set('Vis_learn','n_quantile',str(options.n_quantile))
    Config.set('Vis_learn','min_scale',str(options.min_scale))
    Config.set('Vis_learn','max_scale',str(options.max_scale))
    Config.set('Vis_learn','min_v',str(options.min_v))
    Config.set('Vis_learn','max_v',str(options.max_v))
    Config.set('Vis_learn','num_bin',str(options.num_bin))
    Config.set('Vis_learn','auto_v',options.auto_v)
    Config.set('Vis_learn','mode_pre_img',options.mode_pre_img)
    Config.set('Visual','channel', str(options.channel))
    Config.set('Visual','dim_img',str(options.dim_img))
    Config.set('Visual','visualize_model',options.visualize_model)
    Config.set('Visual','algo_redu',options.algo_redu)
    Config.set('Visual','reduc_perle',str (options.reduc_perle))
    Config.set('Visual','reduc_ini',options.reduc_ini)
    Config.set('Visual','type_emb',options.type_emb)
    Config.set('Visual','imp_fea',options.imp_fea)
    Config.set('Visual','label_emb', str( options.label_emb))
    Config.set('Visual','emb_data',options.emb_data)
    Config.set('Visual','type_bin',options.type_bin)
    Config.set('Visual','perlexity_neighbor',str(options.perlexity_neighbor))
    Config.set('Visual','n_components_emb',str(options.n_components_emb))
    Config.set('Visual','shape_drawn',options.shape_drawn)
    Config.set('Visual','fig_size',str(options.fig_size))
    Config.set('Visual','point_size',str(options.point_size))
    Config.set('Visual','colormap',options.colormap)
    Config.set('Visual','margin',str(options.margin))
    Config.set('Visual','alpha_v',str(options.alpha_v))
    Config.set('Visual','recreate_img',str( options.recreate_img ))

    #save configuration file
    with open(configfile, 'w') as f:
        Config.write(f)

def validation_para(options):
    '''
    check whether parameters are valid or not
    if there is any error, stop the running and show errors
        # refer https://stackoverflow.com/questions/287871/#print-in-terminal-with-colors to find a color to display the error
    '''
    valid = 1 #if there is any error, then set valid=0 and stopping app
    
    #check para with categories type
    if options.data_name == '':
        print  ( utils.textcolor_display('--data_name')+" is empty! Please specify the dataset ")
        valid = 0

    if not (options.type_run in ['vis','visual','learn','predict','config']):
        print  ('you entered --type_run=' + str(options.type_run))
        print  (utils.textcolor_display('--type_run') + " only supports 'vis','visual','learn','predict','config'")
        valid = 0
    
    if not (options.search_already in ['y','n']):
        print  ('you entered --search_already=' + str(options.search_already))
        print  (utils.textcolor_display('--search_already') + " only supports 'y','n'")
        valid = 0
    
    if not (options.debug in ['y','n']):
        print  ('you entered --debug=' + str(options.debug))
        print  (utils.textcolor_display('--debug') + " only supports 'y','n'")
        valid = 0

    if not (options.check in ['y','n']):
        print  ('you entered --check=' + str(options.check))
        print  (utils.textcolor_display('--check') + " only supports 'y','n'")
        valid = 0

    if not (options.save_avg_run in ['y','n']):
        print  ('you entered --save_avg_run=' + str(options.save_avg_run))
        print  (utils.textcolor_display('--save_avg_run') + " only supports 'y','n'")
        valid = 0
    
    if not (options.save_labels in ['y','n']):
        print  ('you entered --save_labels=' + str(options.save_labels))
        print  (utils.textcolor_display('--save_labels') + " only supports 'y','n'")
        valid = 0

    if not (options.save_d in ['y','n']):
        print  ('you entered --save_d=' + str(options.save_d))
        print  (utils.textcolor_display('--save_d') + " only supports 'y','n'")
        valid = 0

    if not (options.save_w in ['y','n']):
        print  ('you entered --save_w=' + str(options.save_w))
        print  (utils.textcolor_display('--save_w') + " only supports 'y','n'")
        valid = 0
    
    if not (options.save_rf in ['y','n']):
        print  ('you entered --save_rf=' + str(options.save_rf))
        print  (utils.textcolor_display('--save_rf') + " only supports 'y','n'")
        valid = 0

    if not (options.save_w in ['y','n']):
        print  ('you entered --save_w=' + str(options.save_w))
        print  (utils.textcolor_display('--save_w') + " only supports 'y','n'")
        valid = 0

    if not (options.save_para in ['y','n']):
        print  ('you entered --save_para=' + str(options.save_para))
        print  (utils.textcolor_display('--save_para') + " only supports 'y','n'")
        valid = 0
    
    if not (options.save_entire_w in ['y','n']):
        print  ('you entered --save_entire_w=' + str(options.save_entire_w))
        print  (utils.textcolor_display('--save_entire_w') + " only supports 'y','n'")
        valid = 0
    
    if not (options.save_folds in ['y','n']):
        print  ('you entered --save_folds=' + str(options.save_folds))
        print  (utils.textcolor_display('--save_folds') + " only supports 'y','n'")
        valid = 0
    
    if not (options.sound_fini in ['y','n']):
        print  ('you entered --sound_fini=' + str(options.sound_fini))
        print  ( utils.textcolor_display('--sound_fini') + " only supports 'y','n'")
        valid = 0
    
    if not (options.test_ext in ['y','n']):
        print  ('you entered --test_ext=' + str(options.test_ext))
        print  (utils.textcolor_display('--test_ext') + " only supports 'y','n'")
        valid = 0
    
    if (options.ext_data not in [''] and options.test_ext in ['n']):
        print  ('you entered --ext_data=' + str(options.ext_data) + ' and test_ext=' + str(options.test_ext))
        print  (utils.textcolor_display('--ext_data') + " is used if " + utils.textcolor_display('--test_ext')+ "='y'")
        valid = 0
    
    if not (options.preprocess_img in ['resnet50','vgg16','vgg19','none']):
        print  ('you entered --preprocess_img=' + str(options.preprocess_img))
        print  (utils.textcolor_display('--preprocess_img') + " only supports resnet50/vgg16/vgg19/none")
        valid = 0
    
    if not (options.padding in ['y','n']):
        print  ('you entered --padding=' + str(options.padding))
        print  (utils.textcolor_display('--padding') + " only supports 'y','n'")
        valid = 0
    
    #if not (options.seed_ml in ['y','n']):
    #    #print  ('you entered --seed_ml=' + str(options.seed_ml)
    #    #print  (utils.textcolor_display('--seed_ml') + " only supports 'y','n'"
    #    valid = 0
    
    if not (options.optimizer in ['sgd','adam','Adamax','RMSprop','Adagrad','Adadelta','Nadam']):
        print  ('you entered --optimizer=' + str(options.optimizer))
        print  (utils.textcolor_display('--optimizer') + " only supports sgd/adam/Adamax/RMSprop/Adagrad/Adadelta/Nadam")
        valid = 0



    if not (options.e_stop_consec in ['consec', 'normal']):
        print  ('you entered --e_stop_consec=' + str(options.e_stop_consec))
        print  (utils.textcolor_display('--e_stop_consec') + " only supports consec/normal")
        valid = 0
    
    if not (options.svm_kernel in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']):
        print  ('you entered --svm_kernel=' + str(options.svm_kernel))
        print  (utils.textcolor_display('--svm_kernel') + " only supports linear, poly, rbf, sigmoid, precomputed")
        valid = 0
    
    if not (options.del0 in ['y','n']):
        print  ('you entered --del0=' + str(options.del0))
        print  (utils.textcolor_display('--del0') + " only supports 'y','n'")
        valid = 0
    
    if not ( (options.scale_mode in ['mms','none','','qtf','qtnf','ln','ptf']) or options.scale_mode[0:3] == 'log'):
        print  ('you entered --scale_mode=' + str(options.scale_mode))
        print  (utils.textcolor_display('--scale_mode') + " only supports 'mms' (minmaxscaler),'none','','qtf'(quantiletransformer-uniform distribution),'qtnf'(quantiletransformer-normal),'ln','ptf, and log base n (e.g. log2, log4,...)")
        valid = 0
    
    if not (options.auto_v in ['y','n']):
        print  ('you entered --auto_v=' + str(options.auto_v))
        print  (utils.textcolor_display('--auto_v') + " only supports 'y','n'")
        valid = 0

    if not (options.mode_pre_img in ['caffe','tf' ]):
        print  ('you entered --mode_pre_img=' + str(options.mode_pre_img))
        print  (utils.textcolor_display('--mode_pre_img') + " only supports caffe/tf ")
        valid = 0

    if not (options.type_emb in ['raw','bin','fill','fills','zfill','zfills','tsne','megaman','lle','isomap','mds','se','pca','rd_pro','lda','nmf','minisom']):
        print  ('you entered --type_emb=' + str(options.type_emb))
        print  (utils.textcolor_display('--type_emb') + " only supports 'raw','bin','fill','fills','zfill','zfills','tsne','megaman','lle','isomap','mds','se','pca','rd_pro','lda','nmf','minisom'")
        valid = 0

    if not (options.visualize_model in ['y','n']):
        print  ('you entered --visualize_model=' + str(options.visualize_model))
        print  (utils.textcolor_display('--visualize_model') + " only supports 'y','n'")
        valid = 0
    
    if not (options.imp_fea in ['none','rf','avg','','max']):
        print  ('you entered --imp_fea=' + str(options.imp_fea))
        print  (utils.textcolor_display('--imp_fea') + " only supports '','none','rf','avg','max'")
        valid = 0
    
    if not (options.emb_data in ['','o']):
        print  ('you entered --emb_data=' + str(options.visualize_model))
        print  (utils.textcolor_display('--emb_data') + " only supports '','o' (orginal data)")
        valid = 0

    if not (options.type_bin in ['spb','pr','eqw','eqf','']):
        print  ('you entered --type_bin=' + str(options.type_bin))
        print  (utils.textcolor_display('--type_bin') + " only supports 'spb','pr','eqw','eqf' or skip this option")
        valid = 0
     

    if not (options.model in ['svm_model','rf_model','dtc_model','gbc_model','knn_model','model_cnn','pretrained','fc_model','model_lstm','model_mlp','model_vgglike','model_cnn1d']):
        print  ('you entered --model=' + str(options.model))
        print  (utils.textcolor_display('--model') + " only supports 'svm_model','rf_model','dtc_model','gbc_model','knn_model','model_cnn','pretrained','fc_model','model_lstm','model_mlp','model_vgglike','model_cnn1d'")
        valid = 0

    #check validation with logic contrainst
    if options.type_run in ['vis','visual'] and options.type_emb in ['raw','bin']:
        print  (utils.textcolor_display('--type_emb') + " = " + options.type_emb +". Visualization cannot be made with 1D! Please select Fill-up or t-SNE or another manifold learning!")
        valid = 0
    
    if options.test_size > 1 or options.test_size < 0:
        print  ('you entered --test_size=' + str(options.test_size))
        print  ( utils.textcolor_display('--test_size') + ' must be from 0 to 1')
        valid = 0
    
    if options.type_run in ['predict'] and options.model not in ['pretrained']:
        print  (utils.textcolor_display('--model') + ' should be  "pretrained" if --type_run=predict')
        valid = 0
        
    if options.model in ['pretrained'] and options.pretrained_w_path == '':       
        print  (utils.textcolor_display('--pretrained_w_path') + ' must be provided if --model=pretrained' )   
        valid = 0 
   
    if options.orderf_fill in ['high'] and not(options.type_emb in ['fills','zfills']):
        print  ('you entered --orderf_fill=' + str(options.orderf_fill) + ' and --type_emb='+ str(options.type_emb))
        print  (utils.textcolor_display('--orderf_fill='+ str(options.orderf_fill)) + " only supports with --type_emb in [fills,zfills]")
        valid = 0
   
    if options.type_bin != '' and options.type_emb in ['raw']:
        print  ('you entered --type_emb=' + str(options.type_emb) + ' and --type_bin='+ str(options.type_bin))
        print  (utils.textcolor_display('Please remove --type_bin if you want to use --type_emb='+ str(options.type_emb) + '. If you want to use --type_bin, please do not set --type_emb=raw'))
        valid = 0
   
    if valid == 0:
        exit()

def naming_folder (options, search_res = 'y'):
    '''
    set prefix for folder name
    '''
    path_write_parentfolder_img = []
    #if options.original_data_folder[0] == "/": #use absolute path
    #    path_read =  options.original_data_folder 
    #else: #relative path
    #    path_read = os.path.join('.', options.original_data_folder) 
    
    #pre_name: prefix name of folder       
    if not (options.algo_redu in ['','none']):  #if use reducing dimension
        if options.algo_redu == 'rd_pro':
            if options.rd_pr_seed != "None":
                pre_name = options.data_name + options.algo_redu + str(options.new_dim) + '_'+ str(options.reduc_perle)   + '_s' + str(options.rd_pr_seed)
            else:
                pre_name = options.data_name + options.algo_redu + str(options.new_dim) + '_'+ str(options.reduc_perle)  
        else:
            pre_name = options.data_name + options.algo_redu + str(options.new_dim) + '_'+ str(options.reduc_perle)   
    else:
        pre_name = options.data_name
    
    if options.del0  == 'y': #if remove columns which own all = 0
        pre_name = pre_name + '_del0'

    if not(options.orderf_fill in ["none"]): #set name for features ordered randomly
        if options.type_emb in ['raw','fill','fills','zfill','zfills']:
            if options.orderf_fill in ['high']:
                pre_name = pre_name + 'highf' 
            else:
                pre_name = pre_name + 'rnd' + str(options.orderf_fill)
        else:
            print  ( utils.textcolor_display("use of features ordered only supports on raw','fill','fills', 'zfill','zfills'"))
            exit()
   


    if options.type_bin == 'pr': #if select bin=binary black/white
        options.num_bin = 2

    if options.type_emb == 'raw':      #raw data  
        if options.scale_mode == '':
            pre_name = pre_name + "_" + options.type_emb
        else:
            pre_name = pre_name + '_' + str(options.type_emb) + '_'            
            if options.scale_mode in ['qtf','qtnf']:     
                pre_name = pre_name + str(options.scale_mode) + str(options.n_quantile) 
            elif options.scale_mode  == 'mms':     
                pre_name = pre_name + str(options.scale_mode) + '_isc'+str(options.min_scale) + '_asc'+str(options.max_scale)
            elif not (options.scale_mode in ['','none']):             
                pre_name = pre_name + options.scale_mode
            
    
    elif options.type_emb == 'bin': #bin: spb (manual bins), eqw (equal width bins), pr (binary: 2 bins), eqf
        
        pre_name = pre_name + '_' + str(options.type_emb) + options.type_bin 
        path_write_parentfolder_img = os.path.join('.', options.parent_folder_img, pre_name)
        pre_name =pre_name+'_nb'+str(options.num_bin) + '_au'+str(options.auto_v) + '_'+str(options.min_v) + '_'+str(options.max_v)
             
        if options.scale_mode in ['qtf','qtnf']:     
            pre_name = pre_name + str(options.scale_mode) + str(options.n_quantile)         
        elif options.scale_mode  == 'mms':     
            pre_name = pre_name + str(options.scale_mode) + '_isc'+str(options.min_scale) + '_asc'+str(options.max_scale)
        elif not (options.scale_mode in ['','none']):             
            pre_name = pre_name + options.scale_mode

        if not os.path.exists(path_write_parentfolder_img):
            #print  (path_write_parentfolder_img + ' does not exist, this folder will be created...' 
            os.makedirs(os.path.join(path_write_parentfolder_img)) 

    elif options.type_emb in  ['fill','zfill']: 
        #fill, create images only one time
        # set the names of folders for outputs    
        shape_draw=''  
        if options.shape_drawn == ',': #',' means pixel in the graph
            shape_draw='pix'
        pre_name = pre_name + '_' + str(options.type_emb) + '_' + shape_draw + '_r' + str(options.fig_size) + 'p'+ str(options.point_size) + options.type_bin + "m" + str(options.margin) +'a'+str(options.alpha_v)  + str(options.colormap) + 'bi' + str(options.num_bin)+'_'+str(options.min_v) + '_'+str(options.max_v)
        if options.cmap_vmin != 0 or options.cmap_vmax != 1:
            pre_name = pre_name + 'cmv' + str(options.cmap_vmin) +'_' + str(options.cmap_vmax)
        path_write_parentfolder_img = os.path.join('.', options.parent_folder_img, pre_name)
        if not os.path.exists(path_write_parentfolder_img):
            print  (path_write_parentfolder_img + ' does not exist, this folder will be created...' )
            os.makedirs(os.path.join(path_write_parentfolder_img))     
 
    else: #if Fills (eqw) OR using other embeddings: tsne, LLE, isomap,..., set folder and variable, names
        #parameters for manifold: perplexsity in tsne is number of neigbors in others
        learning_rate_x = options.lr_visual
        n_iter_x = options.iter_visual
        perplexity_x = options.perlexity_neighbor  
        shape_draw=''  
        if options.shape_drawn == ',':
            shape_draw='pix'  #pixel

        #add pre_fix for image name
        if options.type_emb in  ['fills','zfills']:            
            pre_name = pre_name+'_'+str(options.type_emb)  + str(shape_draw) + '_r' + str(options.fig_size) + 'p'+ str(options.point_size) + options.type_bin + "m" + str(options.margin) +'a'+str(options.alpha_v) +  str(options.colormap)
        else: #others: tsne, lle, pca, lda ...            
            if options.imp_fea in ['none']:  #if use RF to arrange important features to a front position
                insert_named = ''      
            else:
                insert_named =  str(options.imp_fea)

            if options.type_emb == 'lda':
                #if use supervised dimensional reduction, so determine which label used.
                pre_name = pre_name+'_'+str(options.type_emb)+ options.eigen_solver  + str(options.label_emb) +str(options.emb_data) + insert_named + '_p'+str(perplexity_x) + 'l'+str(learning_rate_x) + 'i'+str(n_iter_x) + shape_draw + '_r' + str(options.fig_size) + 'p'+ str(options.point_size) + options.type_bin + "m" + str(options.margin) +'a'+str(options.alpha_v)  + str(options.colormap)
       
            else:
                if options.label_visual == "": #naming for other manifolds
                    pre_name = pre_name+'_'+str(options.type_emb) +str(options.emb_data) + insert_named + '_p'+str(perplexity_x) + 'l'+str(learning_rate_x) + 'i'+str(n_iter_x) + shape_draw + '_r' + str(options.fig_size) + 'p'+ str(options.point_size) + options.type_bin + "m" + str(options.margin) +'a'+str(options.alpha_v) +  str(options.colormap)
                else: 
                    pre_name = pre_name+'_'+str(options.type_emb) +str(options.emb_data) + str(options.label_visual)+ insert_named+ '_p'+str(perplexity_x) + 'l'+str(learning_rate_x) + 'i'+str(n_iter_x) + shape_draw + '_r' + str(options.fig_size) + 'p'+ str(options.point_size) + options.type_bin + "m" + str(options.margin) +'a'+str(options.alpha_v) +  str(options.colormap)
            
        pre_name =pre_name+'_nb'+str(options.num_bin) + '_au'+str(options.auto_v) + '_'+str(options.min_v) + '_'+str(options.max_v) 
        
        if options.cmap_vmin != 0 or options.cmap_vmax != 1:
            pre_name = pre_name + 'cmv' + str(options.cmap_vmin) +'_' + str(options.cmap_vmax)

        if options.type_emb == 'lle':
            pre_name = pre_name + str(options.method_lle)


    
        if options.scale_mode in ['qtf','qtnf']:     
            pre_name = pre_name + str(options.scale_mode) + str(options.n_quantile)         
        elif options.scale_mode  == 'mms':     
            pre_name = pre_name + str(options.scale_mode) + '_isc'+str(options.min_scale) + '_asc'+str(options.max_scale)
        elif not (options.scale_mode in ['','none']):             
            pre_name = pre_name + options.scale_mode

        path_write_parentfolder_img = os.path.join('.', options.parent_folder_img, pre_name)
        #looks like: cir_p30l100i500/          
        if not os.path.exists(path_write_parentfolder_img):
            print  (path_write_parentfolder_img + ' does not exist, this folder will be created...' )
            os.makedirs(os.path.join(path_write_parentfolder_img))  
    
    res_dir = os.path.join(".", options.parent_folder_results, pre_name) 

    if options.debug == 'y':
        print  (res_dir )

    # check for existence to contain results, if not, then create
    if not os.path.exists(os.path.join(res_dir)):
        print  (res_dir + ' does not exist, this folder will be created to contain results...')
        os.makedirs(os.path.join(res_dir))
    if not os.path.exists(os.path.join(res_dir, 'details')):       
        os.makedirs(os.path.join(res_dir, 'details'))
    if not os.path.exists(os.path.join(res_dir, 'models')):
        os.makedirs(os.path.join(res_dir, 'models'))      
    
    #check whether this experiment had done before
    if search_res == 'y':
        if options.search_already == 'y':
            prefix_search = name_log_final(options, n_time_text= '*')        
            
            list_file = utils.find_files(prefix_search+'file*'+str(options.suff_fini)+'*',res_dir)
            
            if len(list_file)>0:
                print  ('the result(s) existed at:')
                print  (list_file)
                print  ('If you would like to repeat this experiment, please set ' + utils.textcolor_display('--search_already')+' to n')
                exit()
            else:
                #if options.cudaid not in  ["-1",'']:
                if options.cudaid > -3:
                    #list_file = utils.find_files(prefix_search+'gpu'+ str(options.cudaid)+'file*'+str(options.suff_fini)+'*',res_dir)
                    list_file = utils.find_files(prefix_search+'gpu_file*'+str(options.suff_fini)+'*',res_dir)
                    if len(list_file)>0:
                        #print  ('the result(s) existed at:'
                        #print  (list_file
                        #print  ('If you would like to repeat this experiment, please set ' + utils.textcolor_display('--search_already')+' to n'
                        exit()
    
    return path_write_parentfolder_img, res_dir, pre_name

def name_log_final(arg, n_time_text):
    """ naming for log file
    Args:
        arg (array): arguments input from keybroad
        n_time_text (string) : beginning time
    Returns:
        a string (use for either prefix or suffix for name of log)
    """
    
    if arg.type_run in ['predict']:
        return 'predict' + n_time_text + '_'+ os.path.basename(arg.pretrained_w_path)
         
    prefix=''

    mid="o"+str(arg.num_classes)+arg.optimizer+"_lr"+str(arg.learning_rate)+'de'+str(arg.learning_rate_decay)+"e"+str(arg.epoch)+"_"+str(n_time_text) + 'c' +str(arg.coeff) + 'di' + str(arg.dim_img) + 'ch' + str(arg.channel) #+  'bi' + str(arg.num_bin)+'_'+str(arg.min_v) + '_'+str(arg.max_v)
    
    #check which model will be used
    #model for 2D
    if arg.model in ['model_cnn'] : #Convolution 2D        
        prefix='cnn_' + mid + "_l" + str(arg.numlayercnn_per_maxpool) +"f"+str(arg.numfilters)+"dfc"+str(arg.dropout_fc)+"p"+str(arg.nummaxpool)+"dcnn"+str(arg.dropout_cnn) + "pad" + str(arg.padding)                 
    elif arg.model=="model_mlp": #Multi-Layer Perception
        prefix='mlp_' + mid + "_l" + str(arg.numlayercnn_per_maxpool) +"f"+str(arg.numfilters)+"dfc"+str(arg.dropout_fc)
    elif arg.model=="model_lstm": #Long Short Term Memory networks
        prefix='lstm_' + mid + "_l" + str(arg.numlayercnn_per_maxpool) +"f" + str(arg.numfilters)+"dfc"+str(arg.dropout_fc)
    elif arg.model=="model_vgglike": #Model VGG-like
       prefix='vgg_' + mid  + "dfc"+str(arg.dropout_fc) +"dcnn" + str(arg.dropout_cnn)+ "pad" + str(arg.padding)   
    elif arg.model=="model_cnn1d": #Convolution 1D
        prefix = 'cn1d_' + mid + "_l" + str(arg.numlayercnn_per_maxpool) +"f"+str(arg.numfilters)+"dfc"+str(arg.dropout_fc)+"p"+str(arg.nummaxpool)+"dcnn"+str(arg.dropout_cnn) + "pad" + str(arg.padding) 
    elif arg.model=="fc_model": #FC model 
        prefix='fc_' + mid + "dfc" +str(arg.dropout_fc) 
    #model for 1D
    elif arg.model=="svm_model": #SVM model 
        prefix='svm_' + str(arg.svm_kernel)+str(arg.svm_c)+'_' + mid 
    elif arg.model=="rf_model": #Random forest model 
        if arg.rf_max_depth in [-1,'-1']:
            prefix='rf_' +  str(arg.rf_n_estimators) +'_'+ str(arg.rf_max_features) + '_'+mid 
        else:
            prefix='rf_' +  str(arg.rf_n_estimators) + '_'+ str(arg.rf_max_features)  +'dp'+ str(arg.rf_max_depth)+'_'+mid 
    elif arg.model=="dtc_model": #Random forest model 
        if arg.rf_max_depth in [-1,'-1']:
            prefix='dtc_' + str(arg.rf_max_features) + '_'+mid 
        else:
            prefix='dtc_' + str(arg.rf_max_features)  +'dp'+ str(arg.rf_max_depth)+'_'+mid 

    elif arg.model=="gbc_model": #gbc
        if arg.rf_max_depth in [-1,'-1']:
            prefix='gbc_' +  str(arg.rf_n_estimators) + '_'+ str(arg.rf_max_features)+ '_'+mid 
        else:
            prefix='gbc_' +  str(arg.rf_n_estimators) + '_'+ str(arg.rf_max_features)+ 'dp'+ str(arg.rf_max_depth)+'_'+mid 
    elif arg.model=="knn_model": #knn
        prefix='knn_' +  str(arg.knn_n_neighbors) + '_'+mid 
    else:
        prefix = arg.model +'_'+ mid + "l"+str(arg.numlayercnn_per_maxpool)+"p"+str(arg.nummaxpool)+"f"+str(arg.numfilters)+"dcnn"+str(arg.dropout_cnn)+"dfc"+str(arg.dropout_fc)
    
    #use consec or estop
    if arg.e_stop_consec=="consec":
        prefix = 'estopc' + str(arg.e_stop) + '_' + prefix
    else:
        prefix = 'estop' + str(arg.e_stop) + '_' + prefix
    
    #if use !='bin' or 'raw' --> images, so specify name of preprocessing images
    if arg.type_emb != 'bin' and arg.type_emb != 'raw':
        prefix = str(arg.preprocess_img) + str(arg.mode_pre_img)+'_' + prefix
    
    if arg.test_size not in  [0,1] :
        prefix = 'ts'+str(arg.test_size)+'_' + prefix
    else:
        if arg.n_folds != 10:
            prefix = 'k'+str(arg.n_folds)+'_' + prefix

    #if arg.run_time != 10: # add #times of the experiment
    prefix = 'a'+str(arg.run_time)+'_' + prefix

    if arg.loss_func!='binary_crossentropy':
        prefix = prefix + "los" + arg.loss_func
    
    if arg.whole_run_time != 1: # add #times of the experiment
        prefix = 'all'+str(arg.whole_run_time)+'_' + prefix
    
    if arg.save_w == 1: #if save weights of all models in cv
        prefix = prefix + "weight"

     
    if arg.cudaid > -3:   
        prefix = prefix + "gpu" + str(arg.cudaid)
    
    if arg.batch_size != 16:
        prefix = prefix + "batchsz" + str(arg.batch_size)
    
    if arg.ext_data != '':
        prefix = prefix + '_ex_' + arg.ext_data + '_'
        
    return prefix

def read_data (options, suffix = '', included_labels = 'y' , get_names_of_features = 'n', mode='internal') :
    '''
    Read data
        options (array): provided from users
        suffix (string): default: '', a symbol to distinct 2 types of data '': internal, and 'z' external
        included_labels (y or n): y if read labels
        mode: internal or external
    
    Return data, labels
    
    Notes:
        options used in the function
            options.original_data_folder
            options.data_name
            options.ext_data
            options.orderf_fill
            options.type_emb

    ** orderf_fill used if and only if type_emb == 'fill' or a type of fill
    
    '''
    #read x: data
    v_data_ori = []
    v_labels = []

    #print  ('========reading ' + mode + ' data======='
    if mode == 'internal':        
        path_v_set_x = os.path.join(options.original_data_folder , options.data_name + '_' +  'x.csv')   
    elif mode == 'external':        
        if options.ext_data == '':
            #print  ('with the same pattern name with internal set'
            path_v_set_x = os.path.join(options.original_data_folder , options.data_name + str(suffix) + '_' +  'x.csv') 
        else:
            #print  ('with the different pattern name with internal set'
            path_v_set_x = os.path.join(options.original_data_folder , options.ext_data  + '_' +  'x.csv')   

    if not os.path.exists(path_v_set_x):
        print( utils.textcolor_display (path_v_set_x + ' does not exist!!'))
        exit()   
    v_data_ori = pd.read_csv(path_v_set_x) #rows: features, colulms: samples 
    v_data_ori = v_data_ori.set_index('Unnamed: 0')
    
    if get_names_of_features in ['y','yes']:
        name_features =  list(v_data_ori.index)

    v_data_ori = v_data_ori.values
    if options.orderf_fill != "none": #set name for features ordered randomly
        if options.type_emb in ['raw','fill','fills','zfill','zfills']:
            if options.orderf_fill != 'high':            
                np.random.seed(int(options.orderf_fill))
                np.random.shuffle(v_data_ori) #shuffle features with another order using seed 'orderf_fill'
        else:
            print  ( utils.textcolor_display ("use of features ordered randomly only supports on raw','fill','fills'"))
            exit()

    v_data_ori = np.transpose(v_data_ori) #rows: samples, colulms: features
    v_data_ori = np.stack(v_data_ori)  

    #read y: labels
    if included_labels in ['y']:
        if mode == 'internal':            
            path_v_set_y = os.path.join(options.original_data_folder , options.data_name + '_' +  'y.csv')   
        elif mode == 'external':                
            if options.ext_data == '':
                #print  ('with the same pattern name with internal set'
                path_v_set_y = os.path.join(options.original_data_folder , options.data_name + str(suffix) + '_' +  'y.csv') 
            else:
                #print  ('with the different pattern name with internal set'
                path_v_set_y = os.path.join(options.original_data_folder , options.ext_data  + '_' +  'y.csv')   
        
        if not os.path.exists(path_v_set_y):
            print( utils.textcolor_display (path_v_set_y + ' does not exist!!'))
            exit()       
        #print  ('========reading label======='
        v_labels = pd.read_csv(path_v_set_y)
        v_labels = v_labels.iloc[:,1] 
    
    #print  (v_data_ori.shape
    if get_names_of_features in ['y','yes']:
        return v_data_ori, v_labels, name_features
    else:
        return v_data_ori, v_labels
    
def create_fillup_folder (options,path_write_parentfolder_img,
        cordi_x,cordi_y,data,labels, prefix_img = 'fill', only_visual = 'n'):
    '''
    creating images, saving a folder with fill-up and read images
    arg:
        options (list): options from users
        path_write_parentfolder_img (string): folder path to save images
        cordi_x, cordi_y: coordinates of x and y to range values into a square
        data (array): data to fill-up
        labels (list): labels
        prefix_img: set prefix for naming images
    
    return
        array of values of pixel

    '''
    #check whether images created completely or requirement for creating images from options; if not, then creating images
    #create images if does not exist
   
    if options.fig_size == 0: #if fig_size <=0, use the smallest size which fit the data, generating images of (len_square x len_square)
        fig_v_size =  int(math.ceil(math.sqrt(data.shape[1])))  
    else:
        fig_v_size =  options.fig_size
    #print  ('fig_v_size='+str(fig_v_size)


    if options.auto_v  == 'y': #if auto, adjust automatically min_v and max_v for binning
        #options.min_v = np.min(train_x)
        #options.max_v = np.max(train_x)
        min_v_temp = np.min(data)
        max_v_temp = np.max(data)
    else:
        min_v_temp = options.min_v
        max_v_temp = options.max_v

    if not os.path.exists(path_write_parentfolder_img+'/'+prefix_img + '_' + str(len(labels)-1)+'.png') or options.recreate_img > 0 :
        #labels.to_csv(path_write_parentfolder_img+'/' + prefix_img+  '_y.csv', sep=',')
        mean_spe=(np.mean(data,axis=0))       

        # generate bin breaks
        bins_breaks = vis_data.build_breaks (data=data, type_bin=options.type_bin, num_bin=options.num_bin, max_v= max_v_temp, min_v = min_v_temp)
        
        # generate bins for the sample
        vis_data.generate_image (data = mean_spe, bins_breaks = bins_breaks, colormap = options.colormap, 
            file_name_emb = path_write_parentfolder_img + "/global",size_p = options.point_size, fig_size = fig_v_size,
            marker = options.shape_drawn, alpha_v = options.alpha_v,
            cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
            min_v = min_v_temp, num_bin = options.num_bin,
            cor_x=cordi_x,cor_y=cordi_y)

        for i in range(0, len(labels)):
            print  ('img ' + prefix_img + ' ' + str(i)   )       

            # generate bins for the sample
            vis_data.generate_image (data = data[i], bins_breaks = bins_breaks, colormap = options.colormap, 
                file_name_emb = path_write_parentfolder_img + "/" + prefix_img + "_"+str(i),size_p = options.point_size, fig_size = fig_v_size,
                marker = options.shape_drawn, alpha_v = options.alpha_v,
                cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                min_v = min_v_temp, num_bin = options.num_bin,
                cor_x=cordi_x,cor_y=cordi_y)

    #reading images
    if only_visual in ['n']:
        data = utils.load_img_util(num_sample = len(data), pattern_img= prefix_img,
                path_write = path_write_parentfolder_img , dim_img = options.dim_img, preprocess_img = options.preprocess_img, 
                channel = options.channel,mode_pre_img = options.mode_pre_img)       
    #print  ('data=' + str(data.shape) #check dim of reading
    return data

def reduce_dim (options,prefix_file_log, seed_value, index, train_x, val_x, v_data):
    '''
    reduce dimension of train, validation, test set
    arg:
        options (list): options from users
        prefix_file_log (string): prefix file to save log
        seed_value: value for saving log
        index : value for saving log
        train_x (array): training set
        val_x: validation set
        v_data : test set for external validation
    
    return
        data after reducing dimension

    '''
    if options.type_emb in ['fill','zfill']:
        print  ( utils.textcolor_display ('options.type_emb=fill is not supported!! for options.algo_redu=' +str(options.algo_redu) + '. Did you want to select fills?'))
        exit()

    #print  ('reducing dimension with ' + str(options.algo_redu)
    f=open(prefix_file_log+"_sum.txt",'a')          
    title_cols = np.array([["seed",seed_value," fold",index+1,"/",options.n_folds,"+++++++++++","seed",seed_value," fold",index+1,"/",options.n_folds,"+++++++++++","seed",seed_value," fold",index+1,"/",options.n_folds]])
    np.savetxt(f,title_cols, fmt="%s",delimiter="")             
    title_cols = np.array([["before reducing, tr_mi/me/ma="+ str(np.min(train_x)) + '/' + str(np.mean(train_x))+ '/'+ str(np.max(train_x))+", va_mi/me/ma="+ str(np.min(val_x)) + '/' + str(np.mean(val_x))+ '/'+str(np.max(val_x)) ]])           
    np.savetxt(f,title_cols, fmt="%s",delimiter="/")    

    t_re = time.time()
    f.close() 
    if options.algo_redu ==  "rd_pro": #if random projection
        if options.rd_pr_seed == "None":
            transformer = random_projection.GaussianRandomProjection(n_components=options.new_dim)
        else:                        
            transformer = random_projection.GaussianRandomProjection(n_components=options.new_dim, random_state = int(options.rd_pr_seed))
        train_x = transformer.fit_transform(train_x)
        #val_x = transformer.transform(val_x)
    elif options.algo_redu == "pca":
        try:
            transformer = PCA(n_components=options.new_dim) #just add from v1.0.30
        except:
            transformer = PCA(n_components=options.new_dim, svd_solverstr = 'auto')
        train_x = transformer.fit_transform(train_x)
        #val_x = transformer.transform(val_x)
    
    elif options.algo_redu == 'fa': # FeatureAgglomeration
        transformer = FeatureAgglomeration(n_clusters=options.new_dim)
        train_x = transformer.fit_transform(train_x)
        #val_x = transformer.transform(val_x)
    
    else:
        print  (utils.textcolor_display (' this algo_redu ' + str(options.algo_redu) + ' is not supported!'))
        exit()   

    #tranform validation and test set
    val_x = transformer.transform(val_x)
    if options.test_ext == 'y':  
        v_data = transformer.transform(v_data)

    f=open(prefix_file_log+"_sum.txt",'a')    
    text_s = "after reducing, tr_mi/me/ma=" 
    text_s = text_s + str(np.min(train_x)) + '/' + str(np.mean(train_x))+ '/'+ str(np.max(train_x)) 
    text_s = text_s + ", va_mi/me/ma="+ str(np.min(val_x)) + '/' + str(np.mean(val_x))+ '/'+str(np.max(val_x)) 
    text_s = text_s + ', time_reduce=' +str(time.time()- t_re)
    title_cols = np.array([[text_s]])           
    np.savetxt(f,title_cols, fmt="%s",delimiter="/")            
    f.close()          

    return  train_x, val_x, v_data

def multiclass_roc_auc_score(y_test, y_pred, average="micro"):
    from sklearn.preprocessing import LabelBinarizer

    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)

def visual_section (options,  train_x,val_x, train_y, val_y, v_data, v_labels,train_x_original,
          prefix_file_log='', path_write_parentfolder_img='', cordi_x=[],cordi_y=[],train_test_whole = 'n',seed_value=1, index=1):
    '''
    visual section, generate images with different algorithms, and return pixels of the set of images
        using trainning set to learn and build coordinates, then generating images for both trainning and test sets
    arg:
        options (list): options from users
        path_write_parentfolder_img (string): prefix file to save log
        seed_value: use for log
        train_x, train_y: training set
        val_x, val_y : validation set
        v_data, v_labels: test set (external)
        train_x_original: original data before transformation, use for visualizing using manifolds learning
        cordi_x, cordi_y: coordinates for fill-up    
        train_test_whole : create images for train and test on whole dataset (for naming folders)

    return    train_x,val_x,v_data

    Notes: options used in the function:
        test_size
        n_folds
        fig_size
        auto_v
        emb_data
        label_visual
        n_components_emb
        perlexity_neighbor
        iter_visual
        ini_visual
        lr_visual
        imp_fea
        shape_drawn
        point_size
        type_bin
        margin
        alpha_v        
        colormap
        num_bin
        cmap_vmin
        cmap_vmax
        file_name

    '''
    t_img = time.time()
    # ++++++++++++ start to check folders where contain images ++++++++++++
    path_write= os.path.join(path_write_parentfolder_img,'s'+str(seed_value))
    #print(path_write)
    #looks like: ./images/cir_p30l100i500/s1/        
    if not os.path.exists(path_write):
        #print(path_write + ' does not exist, this folder will be created...')
        os.makedirs(os.path.join(path_write))    

    if options.test_size in [0, 1] : #if use cross validation
        #folder below1
        path_write= os.path.join(path_write_parentfolder_img,'s'+str(seed_value),"nfold"+str(options.n_folds))
        #looks like: ./images/cir_p30l100i500/s1/nfold10/   
        #print(path_write)  
        if not os.path.exists(path_write):
            #print(path_write + ' does not exist, this folder will be created...')
            os.makedirs(os.path.join(path_write))   
        
        #folder below2
        if train_test_whole in ['y','yes']:
            path_write= os.path.join(path_write_parentfolder_img,'s'+str(seed_value),"nfold"+str(options.n_folds),"whole")   
        else:
            path_write= os.path.join(path_write_parentfolder_img,'s'+str(seed_value),"nfold"+str(options.n_folds),"k"+str(index+1))   
            #looks like: ./images/cir_p30l100i500/s1/nfold10/k1 
        #print(path_write)   
        if not os.path.exists(os.path.join(path_write)):
            #print(path_write + ' does not exist, this folder will be created...')
            os.makedirs(os.path.join(path_write))    

    else: #if use holdout validation
        #folder below1
        path_write= os.path.join(path_write_parentfolder_img,'s'+str(seed_value),"ts"+str(options.test_size))
        #looks like: ./images/cir_p30l100i500/s1/ts0.4/   
        #print(path_write)  
        if not os.path.exists(path_write):
            #print(path_write + ' does not exist, this folder will be created...')
            os.makedirs(os.path.join(path_write))   

    if options.fig_size == 0: #if fig_size <=0, use the smallest size which fit the data, generating images of (len_square x len_square)
        fig_v_size =  int(math.ceil(math.sqrt(train_x.shape[1])))  
    else:
        fig_v_size =  options.fig_size
    # !=!=!=!=!=!= end to check folders where contain images !=!=!=!=!=!=

    if options.auto_v  == 'y': #if auto, adjust automatically min_v and max_v for binning
        #options.min_v = np.min(train_x)
        #options.max_v = np.max(train_x)
        min_v_temp = np.min(train_x)
        max_v_temp = np.max(train_x)
    else:
        min_v_temp = options.min_v
        max_v_temp = options.max_v
    
    #generate bin breaks from training set
    bins_breaks = vis_data.build_breaks (data = train_x, type_bin=options.type_bin, num_bin=options.num_bin, max_v= max_v_temp, min_v = min_v_temp)
        
    if options.type_emb in ['fills','zfills'] and options.type_bin != 'pr': #fills: use fill-up with bins created from train-set with scale, should be used with type_bin='eqw'
        #run embbeding to create images if images do not exist     
                
        if (not os.path.exists(path_write+"/val_" + str(len(val_y)-1) + ".png")) or options.recreate_img > 0 :
            #creating images if the sets of images are not completed OR requirement for creating from options.

            mean_spe=(np.mean(train_x,axis=0))     

            if options.orderf_fill == 'high':  
                #X = [["a",2], ["b",3], ["c",3], ["d",3], ["e",2], ["f",4], ["g",5], ["h",1],[ "i",1]]
                #Y = [ 0,   1,   1,    0,   1,   2,   2,   0,   1]
                #Z = [x for _,x in sorted(zip(Y,X))]
                #print(Z)
                #indices_highest_lowest = np.argsort(mean_spe,axis=-1)

                vis_data.generate_image (data = [x for _,x in sorted(zip(mean_spe,mean_spe))], bins_breaks = bins_breaks, colormap = options.colormap, 
                    file_name_emb = path_write + "/global",size_p = options.point_size, fig_size = fig_v_size,
                    marker = options.shape_drawn, alpha_v = options.alpha_v,
                    cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                    min_v = min_v_temp, num_bin = options.num_bin,
                    cor_x=cordi_x,cor_y=cordi_y)

                for i in range(0, len(train_y)):
                    print  ('img fillup-high train ' + str(i) )   
                    #print  (mean_spe_high)
                    #data_or =   train_x[i]
                    #print (train_x[i])
                    #train_high = [x for _,x in sorted(zip(mean_spe,data_or))]    
                    #print (train_high)   
                    vis_data.generate_image (data = [x for _,x in sorted(zip(mean_spe,train_x[i]))], bins_breaks = bins_breaks, colormap = options.colormap, 
                        file_name_emb = path_write + "/train_" + str(i),size_p = options.point_size, fig_size = fig_v_size,
                        marker = options.shape_drawn, alpha_v = options.alpha_v,
                        cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                        min_v = min_v_temp, num_bin = options.num_bin,
                        cor_x=cordi_x,cor_y=cordi_y)             
              
                
                
                for i in range(0, len(val_y)):
                    print  ('img fillup-high val ' + str(i))    
                    vis_data.generate_image (data = [x for _,x in sorted(zip(mean_spe,val_x[i]))], bins_breaks = bins_breaks, colormap = options.colormap, 
                        file_name_emb = path_write + "/val_" + str(i),size_p = options.point_size, fig_size = fig_v_size,
                        marker = options.shape_drawn, alpha_v = options.alpha_v,
                        cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                        min_v = min_v_temp, num_bin = options.num_bin,
                        cor_x=cordi_x,cor_y=cordi_y)     
            
            else:

                
                vis_data.generate_image (data = mean_spe, bins_breaks = bins_breaks, colormap = options.colormap, 
                        file_name_emb = path_write + "/global_train_mean",size_p = options.point_size, fig_size = fig_v_size,
                        marker = options.shape_drawn, alpha_v = options.alpha_v,
                        cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                        min_v = min_v_temp, num_bin = options.num_bin,
                        cor_x=cordi_x,cor_y=cordi_y) 

                for i in range(0, len(train_y)):
                    #print  ('img fillup train ' + str(i)           

                    vis_data.generate_image (data = train_x[i], bins_breaks = bins_breaks, colormap = options.colormap, 
                        file_name_emb = path_write + "/train_" + str(i),size_p = options.point_size, fig_size = fig_v_size,
                        marker = options.shape_drawn, alpha_v = options.alpha_v,
                        cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                        min_v = min_v_temp, num_bin = options.num_bin,
                        cor_x=cordi_x,cor_y=cordi_y)                 
                
                for i in range(0, len(val_y)):
                    #print  ('img fillup val ' + str(i)      

                    vis_data.generate_image (data = val_x[i], bins_breaks = bins_breaks, colormap = options.colormap, 
                        file_name_emb = path_write + "/val_" + str(i),size_p = options.point_size, fig_size = fig_v_size,
                        marker = options.shape_drawn, alpha_v = options.alpha_v,
                        cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                        min_v = min_v_temp, num_bin = options.num_bin,
                        cor_x=cordi_x,cor_y=cordi_y) 

            train_y.to_csv(path_write+'/train_y.csv', sep=',')  
            val_y.to_csv(path_write+'/val_y.csv', sep=',')    

        
        if options.test_ext == 'y':
            if (not os.path.exists(path_write+"/testval_" + str(len(v_labels)-1) + ".png")) or options.recreate_img > 0 :
                if options.orderf_fill == 'high':  
                    #concatenate trainning and test set
                    #all_data = np.concatenate((train_x,val_x))    
                    #mean_spe_whole =(np.mean(all_data,axis=0))         

                    for i in range(0, len(v_labels)):
                        #print  ('img fillup testval ' + str(i)       

                        vis_data.generate_image (data = [x for _,x in sorted(zip(mean_spe,v_data[i]))], bins_breaks = bins_breaks, colormap = options.colormap, 
                            file_name_emb = path_write + "/testval_" + str(i),size_p = options.point_size, fig_size = fig_v_size,
                            marker = options.shape_drawn, alpha_v = options.alpha_v,
                            cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                            min_v = min_v_temp, num_bin = options.num_bin,
                            cor_x=cordi_x,cor_y=cordi_y) 
                else:
                    for i in range(0, len(v_labels)):
                        #print  ('img fillup testval ' + str(i))       
 

                        vis_data.generate_image (data = v_data[i], bins_breaks = bins_breaks, colormap = options.colormap, 
                            file_name_emb = path_write + "/testval_" + str(i),size_p = options.point_size, fig_size = fig_v_size,
                            marker = options.shape_drawn, alpha_v = options.alpha_v,
                            cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                            min_v = min_v_temp, num_bin = options.num_bin,
                            cor_x=cordi_x,cor_y=cordi_y) 
                v_labels.to_csv(path_write+'/test_y.csv', sep=',')             

                    
    elif options.type_emb in  ['tsne','megaman','lle','isomap','mds','se','pca','rd_pro','lda','nmf','minisom'] : #if use other embeddings (not fill), then now embedding!
                
        # ++++++++++++ start to begin embedding ++++++++++++
        # includes 2 steps: Step 1: embedding if does not exist, Step 2: read images

        #run embbeding to create images if images do not exist               
        if not os.path.exists(path_write+"/val_" + str(len(val_y)-1) + ".png") or options.recreate_img > 0 : 
            #creating images if the sets of images are not completed OR requirement for creating from options.
            #print  (path_write + ' does not contain needed images, these images will be created...'
            #print  ('Creating images based on ' + str(options.type_emb) + ' ... ============= '
        
            #compute cordinates of points in maps use either tsne/lle/isomap/mds/se
            #print  ('building cordinates of points based on ' + str(options.type_emb)                   
            if options.emb_data != '': #if use original for embedding
                input_x = np.transpose(train_x_original)
            else:                                                        
                input_x = np.transpose(train_x)                        

            if options.type_emb =='tsne':

                if  options.label_visual != "": #tsne according to label
                    if int(options.label_visual) == 1 or int(options.label_visual) ==0:                                
                        input_x_filter = []
                        ##print  (train_y.shape
                    
                        train_y1= train_y        
                        #reset index to range into folds
                        train_y1=train_y1.reset_index()     
                        train_y1=train_y1.drop('index', 1)
                        ##print  (train_y1
                        for i_tsne in range(0,len(train_y1)):
                            ##print  (str(i_tsne)+'_' + str(train_y1.x[i_tsne])
                            if train_y1.x[i_tsne] == int(options.label_visual):
                                input_x_filter.append(train_x[i_tsne])
                        input_x_filter = np.transpose(input_x_filter)
                        X_embedded = TSNE (n_components=options.n_components_emb,perplexity=options.perlexity_neighbor,n_iter=options.iter_visual, init=options.ini_visual,
                            learning_rate=options.lr_visual).fit_transform( input_x_filter )  
                else:
                    X_embedded = TSNE (n_components=options.n_components_emb,perplexity=options.perlexity_neighbor,n_iter=options.iter_visual, init=options.ini_visual,
                        learning_rate=options.lr_visual).fit_transform( input_x )  
                
            elif options.type_emb =='lle':   
                X_embedded = LocallyLinearEmbedding(n_neighbors=options.perlexity_neighbor, n_components=options.n_components_emb,
                    eigen_solver=options.eigen_solver, method=options.method_lle).fit_transform(input_x)
            elif options.type_emb =='isomap': 
                X_embedded = Isomap(n_neighbors=options.perlexity_neighbor, n_components=options.n_components_emb, max_iter=options.iter_visual, 
                    eigen_solver= options.eigen_solver).fit_transform(input_x)
            elif options.type_emb =='nmf': 
                X_embedded = NMF(n_components=options.n_components_emb, max_iter=options.iter_visual).fit_transform(input_x)
                    
            elif options.type_emb =='mds': 
                X_embedded = MDS(n_components=options.n_components_emb, max_iter=options.iter_visual,n_init=1).fit_transform(input_x)
            elif options.type_emb =='se': 
                if options.eigen_solver in ['none','None']:
                    X_embedded = SpectralEmbedding(n_components=options.n_components_emb,
                        n_neighbors=options.perlexity_neighbor).fit_transform(input_x)
                else:
                    X_embedded = SpectralEmbedding(n_components=options.n_components_emb,
                        n_neighbors=options.perlexity_neighbor, eigen_solver= options.eigen_solver).fit_transform(input_x)
            elif options.type_emb =='pca': 
                X_embedded = PCA(n_components=options.n_components_emb).fit_transform(input_x)
            elif options.type_emb =='rd_pro': 
                X_embedded = random_projection.GaussianRandomProjection(n_components=options.n_components_emb).fit_transform(input_x)
            elif options.type_emb =='minisom': 
                from minisom import MiniSom
                #input_x_temp = np.transpose(input_x) 
                #        x : int
                #            x dimension of the SOM.
                #        y : int
                #            y dimension of the SOM.
                #        input_len : int
                #           Number of the elements of the vectors in input.
                #sigma : float, optional (default=1.0)
                #Spread of the neighborhood function, needs to be adequate
                #to the dimensions of the map.
                #(at the iteration t we have sigma(t) = sigma / (1 + t/T)
                #where T is #num_iteration/2)
                #learning_rate, initial learning rate
                #(at the iteration t we have
                #learning_rate(t) = learning_rate / (1 + t/T)
                #where T is #num_iteration/2)
                som = MiniSom(options.fig_size,options.fig_size,input_x.shape[1],
                    sigma=options.perlexity_neighbor,learning_rate=options.lr_visual)                
                som.pca_weights_init(input_x)
                som.train_batch(input_x,options.iter_visual)
                
                X_embedded = np.empty((0,2), int)
                str_x_y = [] #concanate x and y
                for cnt,xx in enumerate(input_x):
                    w = som.winner(xx)
                    #print (w[1])
                    #print (w[0])
                                       
                    X_embedded = np.append(X_embedded,[[w[0],w[1]]],axis=0)
                    str_x_y = np.append(str_x_y, str(w[0]) + '_' +str(w[1]))
                    
                #print('X_embedded=======')
                #print (X_embedded)
            elif options.type_emb =='lda': #supervised dimensional reductions using labels
                #print  ('input_x'
                #print  (input_x.shape

                phylum_inf = pd.read_csv(os.path.join(options.original_data_folder , options.data_name + '_pl.csv'))
                ##print  ('phylum_inf'
                ##print  (phylum_inf.shape
                ##print  (phylum_inf[:,1]
                #phylum_inf = phylum_inf.iloc[:,1] 

                #supported: 'svd' recommended for a large number of features;'eigen';'lsqr'
                if options.eigen_solver in  ['eigen','lsqr']: 
                    lda = LinearDiscriminantAnalysis(n_components=2,solver=options.eigen_solver,shrinkage='auto')
                else:
                    lda = LinearDiscriminantAnalysis(n_components=2,solver=options.eigen_solver)
                #input_x = np.transpose(input_x)
                ##print  (phylum_inf.iloc[:,1] 
                if options.label_emb <= -1:
                    #print  ("please specify --label_emb for lda: 'kingdom=1','phylum=2','class=3','order=4','family=5','genus=6'"
                    exit
                else:
                    phylum_inf = phylum_inf.iloc[:,options.label_emb ]                          

                X_embedded = lda.fit(input_x, phylum_inf).transform(input_x)
                #X_embedded = LinearDiscriminantAnalysis(
                    #    n_components=options.n_components_emb).fit_transform(np.transpose(input_x),train_y)                           
                #print  (X_embedded
                ##print  (X_embedded.shape
            
            else:
                print  ( utils.textcolor_display (options.type_emb + ' is not supported now!'))
                exit()
    
            #+++++++++ CREATE IMAGE BASED ON EMBEDDING +++++++++                       
            #print  ('creating global map, images for train/val sets and save coordinates....'
            #global map
            mean_spe=(np.mean(input_x,axis=1))    

            if options.imp_fea in ['rf']: 
                ##print  (indices_imp
                ##print  (X_embedded [indices_imp]
                ##print  (mean_spe
                ##print  (mean_spe [indices_imp]
                if options.imp_fea in ['rf']:
                    print  ('find important features')
                forest = RandomForestClassifier(n_estimators=500,max_depth=None,min_samples_split=2)
                forest.fit(train_x, train_y)
                importances = forest.feature_importances_
                indices_imp = np.argsort(importances)
                #indices_imp = np.argsort(importances)[::-1] #reserved order
                f=open(path_write+'/imp.csv','a')
                np.savetxt(f,np.c_[(indices_imp,importances)], fmt="%s",delimiter=",")
                f.close()

                
                bins_breaks = vis_data.build_breaks (data=train_x, type_bin=options.type_bin, num_bin=options.num_bin, max_v= max_v_temp, min_v = min_v_temp)
                vis_data.generate_image (data = mean_spe[indices_imp], bins_breaks = bins_breaks, colormap = options.colormap, 
                        file_name_emb = path_write + "/global_mean_train" ,size_p = options.point_size, fig_size = fig_v_size,
                        marker = options.shape_drawn, alpha_v = options.alpha_v,
                        cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                        min_v = min_v_temp, num_bin = options.num_bin,
                        X_embedded = X_embedded [indices_imp]) 
                np.savetxt (path_write+'/coordinate.csv',X_embedded[indices_imp], delimiter=',')
                                
                #TRAIN: use X_embedded to create images; save labels
                for i in range(0, len(train_x)):                 

                    vis_data.generate_image (data = train_x[i,indices_imp], bins_breaks = bins_breaks, colormap = options.colormap, 
                        file_name_emb = path_write + "/train_" + str(i),size_p = options.point_size, fig_size = fig_v_size,
                        marker = options.shape_drawn, alpha_v = options.alpha_v,
                        cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                        min_v = min_v_temp, num_bin = options.num_bin,
                        X_embedded = X_embedded [indices_imp]) 
                    #print('created img '+"/train_"+str(i))             
                train_y.to_csv(path_write+'/train_y.csv', sep=',')#, index=False)
            
                #TEST: use X_embedded to create images; save labels               
                for i in range(0, len(val_x)):                

                    vis_data.generate_image (data = val_x [i,indices_imp], bins_breaks = bins_breaks, colormap = options.colormap, 
                        file_name_emb = path_write + "/val_" + str(i),size_p = options.point_size, fig_size = fig_v_size,
                        marker = options.shape_drawn, alpha_v = options.alpha_v,
                        cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                        min_v = min_v_temp, num_bin = options.num_bin,
                        X_embedded = X_embedded [indices_imp]) 
                    #print('created img '+"/val_"+str(i))     
                val_y.to_csv(path_write+'/val_y.csv', sep=',')#, index=False)   
           
            elif options.imp_fea in ['avg','max']:
                print (path_write+"/global")
                #print ('---type----'+str(type(X_embedded)))
                
                X_embedded_new, mean_spe_new = overlap_resolution(X_embedded = X_embedded, data=mean_spe, str_x_y = str_x_y, mode = options.imp_fea )
                vis_data.generate_image (data = mean_spe_new, bins_breaks = bins_breaks, colormap = options.colormap, 
                        file_name_emb = path_write + "/global_mean_train",size_p = options.point_size, fig_size = fig_v_size,
                        marker = options.shape_drawn, alpha_v = options.alpha_v,
                        cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                        min_v = min_v_temp, num_bin = options.num_bin,
                        X_embedded = X_embedded_new) 


                np.savetxt (path_write+'/coordinate.csv',X_embedded_new, delimiter=',')
                               
                #TRAIN: use X_embedded to create images; save labels
                for i in range(0, len(train_x)):                       

                    X_embedded_new, train_x_new = overlap_resolution(X_embedded = X_embedded, data= train_x [i,:], str_x_y = str_x_y , mode = options.imp_fea )
                    vis_data.generate_image (data = train_x_new, bins_breaks = bins_breaks, colormap = options.colormap, 
                        file_name_emb = path_write+"/train_"+str(i),size_p = options.point_size, fig_size = fig_v_size,
                        marker = options.shape_drawn, alpha_v = options.alpha_v,
                        cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                        min_v = min_v_temp, num_bin = options.num_bin,
                        X_embedded = X_embedded_new) 
                    #print('created img '+"/train_"+str(i))             
                train_y.to_csv(path_write+'/train_y.csv', sep=',')#, index=False)
            
                #TEST: use X_embedded to create images; save labels               
                for i in range(0, len(val_x)):                

                    X_embedded_new, val_x_new = overlap_resolution(X_embedded = X_embedded, data= val_x [i,:], str_x_y = str_x_y , mode = options.imp_fea )
                    vis_data.generate_image (data = val_x_new, bins_breaks = bins_breaks, colormap = options.colormap, 
                        file_name_emb = path_write+"/val_"+str(i),size_p = options.point_size, fig_size = fig_v_size,
                        marker = options.shape_drawn, alpha_v = options.alpha_v,
                        cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                        min_v = min_v_temp, num_bin = options.num_bin,
                        X_embedded = X_embedded_new) 
                    #print('created img '+"/val_"+str(i))     
                val_y.to_csv(path_write+'/val_y.csv', sep=',')#, index=False)        
            else:
                print (path_write+"/global")
                #print ('---type----'+str(type(X_embedded)))
                vis_data.generate_image (data = mean_spe, bins_breaks = bins_breaks, colormap = options.colormap, 
                        file_name_emb = path_write + "/global_mean_train",size_p = options.point_size, fig_size = fig_v_size,
                        marker = options.shape_drawn, alpha_v = options.alpha_v,
                        cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                        min_v = min_v_temp, num_bin = options.num_bin,
                        X_embedded = X_embedded) 


                np.savetxt (path_write+'/coordinate.csv',X_embedded, delimiter=',')
                               
                #TRAIN: use X_embedded to create images; save labels
                for i in range(0, len(train_x)):                       

                    
                    vis_data.generate_image (data = train_x [i,:], bins_breaks = bins_breaks, colormap = options.colormap, 
                        file_name_emb = path_write+"/train_"+str(i),size_p = options.point_size, fig_size = fig_v_size,
                        marker = options.shape_drawn, alpha_v = options.alpha_v,
                        cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                        min_v = min_v_temp, num_bin = options.num_bin,
                        X_embedded = X_embedded) 
                    #print('created img '+"/train_"+str(i))             
                train_y.to_csv(path_write+'/train_y.csv', sep=',')#, index=False)
            
                #TEST: use X_embedded to create images; save labels               
                for i in range(0, len(val_x)):                


                    vis_data.generate_image (data = val_x [i,:], bins_breaks = bins_breaks, colormap = options.colormap, 
                        file_name_emb = path_write+"/val_"+str(i),size_p = options.point_size, fig_size = fig_v_size,
                        marker = options.shape_drawn, alpha_v = options.alpha_v,
                        cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                        min_v = min_v_temp, num_bin = options.num_bin,
                        X_embedded = X_embedded) 
                    #print('created img '+"/val_"+str(i))     
                val_y.to_csv(path_write+'/val_y.csv', sep=',')#, index=False)                               

        if options.test_ext == 'y':
            if not os.path.exists(path_write+"/testval_" + str(len(v_labels)-1) + ".png") or options.recreate_img > 0 : 
                if options.imp_fea in ['rf']:
                    for i in range(0, len(v_data)):                                  
                        vis_data.generate_image (data = v_data [i,:], bins_breaks = bins_breaks, colormap = options.colormap, 
                            file_name_emb = path_write+"/testval_"+str(i),size_p = options.point_size, fig_size = fig_v_size,
                            marker = options.shape_drawn, alpha_v = options.alpha_v,
                            cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                            min_v = min_v_temp, num_bin = options.num_bin,
                            X_embedded = [indices_imp]) 
                elif options.imp_fea in ['avg','max']:   #use mean for overlapped point
                    for i in range(0, len(v_data)):
                        
                        X_embedded_new, v_data_new = overlap_resolution(X_embedded = X_embedded, data= v_data [i,:], str_x_y = str_x_y , mode = options.imp_fea )
                        #print  ('created img testval ' + str(i)            
                        vis_data.generate_image (data = v_data_new, bins_breaks = bins_breaks, colormap = options.colormap, 
                            file_name_emb = path_write+"/testval_"+str(i),size_p = options.point_size, fig_size = fig_v_size,
                            marker = options.shape_drawn, alpha_v = options.alpha_v,
                            cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                            min_v = min_v_temp, num_bin = options.num_bin,
                            X_embedded = X_embedded_new) 
                else: #normal
                    for i in range(0, len(v_data)):
                        vis_data.generate_image (data = v_data [i,:], bins_breaks = bins_breaks, colormap = options.colormap, 
                            file_name_emb = path_write+"/testval_"+str(i),size_p = options.point_size, fig_size = fig_v_size,
                            marker = options.shape_drawn, alpha_v = options.alpha_v,
                            cmap_vmin = options.cmap_vmin, cmap_vmax = options.cmap_vmax, margin= options.margin, 
                            min_v = min_v_temp, num_bin = options.num_bin,
                            X_embedded = X_embedded) 

       
        #
    else:
        print  ( utils.textcolor_display (str(options.type_emb) + ' is not supported!!') )
        exit() 
    
    
    f=open(prefix_file_log+"_sum.txt",'a')
    title_cols = np.array([["time_read_or/and_create_img=" + str(time.time()- t_img)]])
    np.savetxt(f,title_cols, fmt="%s",delimiter="")
    f.close()   
    #+++++++++ read images for train/val sets +++++++++                
    train_x = utils.load_img_util(num_sample = len(train_y), pattern_img='train',
        path_write = path_write , dim_img = options.dim_img, preprocess_img = options.preprocess_img, 
        channel = options.channel,mode_pre_img = options.mode_pre_img)
    val_x = utils.load_img_util(num_sample = len(val_y), pattern_img='val',
        path_write = path_write , dim_img = options.dim_img, preprocess_img = options.preprocess_img, 
        channel = options.channel,mode_pre_img = options.mode_pre_img)
    if options.test_ext == 'y':
        v_data = utils.load_img_util(num_sample = len(v_labels), pattern_img='testval',
            path_write = path_write , dim_img = options.dim_img, preprocess_img = options.preprocess_img, 
            channel = options.channel,mode_pre_img = options.mode_pre_img)

    return train_x,val_x,v_data

def overlap_resolution(X_embedded,data, str_x_y = [], mode = 'avg'):
    '''
    arg:
        X_embedded
        data
        str_x_y

    return 
        new coordinates after combining
        new data after combining overlapped
    '''
    print ('shape===============')
    print(data.shape)
    print(str_x_y.shape)
    print(X_embedded[:,0].shape)
    print(X_embedded[:,1].shape)

    df = pd.DataFrame({
                    'str_xy' : str_x_y,
                    'x' : X_embedded[:,0],
                    'y': X_embedded[:,1],
                    'v': data,
                })
    df_s = df.sort_values(by=['str_xy'])
    #print('sorted')
    df_s = df_s.reset_index(drop=True)
    
    new_X_embedded = np.empty((0,2), int)
    new_v = []
    new_count1 = []    
    new_sum  = 0
    k = 0
    for i in range(0,len(df_s['str_xy'])):
        #print(df_s['str_xy'][i])                
        if i>0:
            if (df_s['str_xy'][i]==df_s['str_xy'][i-1]):
                k = k + 1
                if mode == 'avg':
                    new_sum = new_sum + df_s ['v'] [i]
                else: # mode == max
                    if new_sum < df_s ['v'] [i]:
                        new_sum = df_s ['v'] [i]
                count_overlapped = count_overlapped + 1
                #print ( str(i)+ '_' + str(df_s ['str_xy'] [i]) + '__v'+ str(df_s ['v'] [i])  + 'overlapped' + str(count_overlapped))
               
            else:                     
                if mode == 'avg':
                    mean_overlapped = float(new_sum) / count_overlapped  
                else:
                    mean_overlapped = new_sum

                new_X_embedded = np.append(new_X_embedded,[[df_s['x'][i-1],df_s['y'][i-1]]],axis=0)
                #print( str(i)+'mean'+str(mean_overlapped))
                new_v = np.append(new_v,mean_overlapped)
                new_count1 = np.append(new_count1,count_overlapped)     
                
                count_overlapped = 1
                new_sum = 0
                #print ( str(i)+ '_' + str(df_s ['str_xy'] [i]) + '__v'+ str(df_s ['v'] [i]) + 't'+str(count_overlapped))
                if mode == 'avg':
                    new_sum = new_sum + df_s ['v'] [i]
                else: # mode == max
                    new_sum = df_s ['v'] [i]
                
        else:
            new_sum = df_s ['v'] [i]
            count_overlapped = 1
            #print ( str(i)+ '_' + str(df_s ['str_xy'] [i]) + '__v'+ str(df_s ['v'] [i]) )        
        
        
    #print(new_sum)
    print('overlapped points:'+str(k))
    if mode == 'avg':
        mean_overlapped = float(new_sum) / count_overlapped  
    else: #mode == max
        mean_overlapped = new_sum
    new_X_embedded = np.append(new_X_embedded,[[df_s['x'][i],df_s['y'][i]]],axis=0)
    new_v = np.append(new_v,mean_overlapped)
    #new_count1 = np.append(new_count1,count_overlapped)  
    return new_X_embedded, new_v

def scaler_section (options,prefix_file_log,seed_value,index,train_x,val_x,v_data):
    '''
    visual section, a procedure to generate images with different algorithms
    arg:
        options (list): options from users
        prefix_file_log (string): prefix file to save log
        seed_value: (#run) use for log
        index: (fold) use for log
        train_x: training set
        val_x: validation set
        v_data: test set       

    '''
    if options.type_emb in ['fill','zfill']:
        print  ( utils.textcolor_display ('options.type_emb=' + options.type_emb + ' is not supported!! for options.scale_mode=' +str(options.scale_mode) + '. Did you want to select fills/zfills?') )
        exit()              

    print  ('before scale:' + str(np.max(train_x)) + '_' + str(np.min(train_x)) + '_' + str(np.max(val_x)) + '_'  + str(np.min(val_x))  )    

    if  options.scale_mode  == 'ln' or options.scale_mode[0:3] == 'log':
        print('tranformation using logarithm')
        if options.scale_mode  == 'ln':
            train_x = np.log(train_x)
            val_x = np.log(val_x)
            if options.test_ext == 'y': 
                v_data = np.log(v_data)
        if options.scale_mode == 'log2':
            train_x = np.log2(train_x)
            val_x = np.log2(val_x)
            if options.test_ext == 'y': 
                v_data = np.log2(v_data)
        else: 
            #print (len(options.scale_mode))
            base_log = int(options.scale_mode[3:len(options.scale_mode)])
            #print (train_x)
            #print (np.log (base_log))
            #print (base_log)
            train_x =  np.log(train_x) / np.log (base_log) 
            val_x =    np.log(val_x) / np.log (base_log)
            if options.test_ext == 'y': 
                v_data = np.log(v_data) / np.log (base_log)

        #resolve with -inf for value of 0
        from numpy import inf
        train_x[train_x == -inf] = 0
        val_x[val_x == -inf] = 0
        train_x = train_x*(-1)
        val_x = val_x*(-1)
        
        if options.test_ext == 'y': 
           v_data[v_data == -inf] = 0
           v_data = v_data*(-1)
    else:
        #select the algorithm for transformation
        if  options.scale_mode in ['minmaxscaler' , 'mms'] :       
            scaler = MinMaxScaler(feature_range=(options.min_scale, options.max_scale)) #rescale value range [options.min_scale , options.max_scale ] 
                #X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
                #X_scaled = X_std * (max - min) + min            
        elif options.scale_mode in [ 'quantiletransformer' , 'qtf']:  
            scaler = QuantileTransformer(n_quantiles=options.n_quantile) 
        elif options.scale_mode in [ 'qtnf']:  
            scaler = QuantileTransformer(n_quantiles=options.n_quantile, output_distribution ='normal')     
        elif options.scale_mode in [ 'ptf' ]:  
            scaler = PowerTransformer()             
        else:
            print( utils.textcolor_display ('this scaler ' + options.scale_mode + ' is not available')    )
            exit()        

        #use training set to learn and apply to validation set 
        train_x = scaler.fit_transform(train_x)
        val_x = scaler.transform(val_x)     
        if options.test_ext == 'y':      #use whole train+val to learn, then create for v_data
            #whole_internal_set = np.concatenate((train_x,val_x))
            #scaler_v.fit(whole_internal_set)
            #v_data = scaler_v.transform(v_data) 
            v_data = scaler.transform(v_data)  

    
    print  ('after scale:' + str(np.max(train_x)) + '_'  + str(np.min(train_x)) + '_' + str(np.max(val_x)) + '_' + str(np.min(val_x))  )                

    #get bins of training set  
    train_hist,train_bin_edges = np.histogram(train_x)
    val_hist0,val_bin_edges0 = np.histogram(val_x)
    val_hist1,val_bin_edges1 = np.histogram(val_x, bins = train_bin_edges)

    #save information on data transformed
    f=open(prefix_file_log+"_sum.txt",'a')          
    title_cols = np.array([["seed",seed_value," fold",index+1,"/",options.n_folds,"+++++++++++","seed",seed_value," fold",index+1,"/",options.n_folds,"+++++++++++","seed",seed_value," fold",index+1,"/",options.n_folds]])
    np.savetxt(f,title_cols, fmt="%s",delimiter="")             
    title_cols = np.array([["after scale, remained information="+ str(sum(val_hist1)) + '/' + str(sum(val_hist0))+'='+ str(float(sum(val_hist1))/sum(val_hist0))]])           
    np.savetxt(f,title_cols, fmt="%s",delimiter="/")
    title_cols = np.array([["max_train="+ str(np.max(train_x)),"min_train="+str(np.min(train_x)),"max_val="+str(np.max(val_x)),"min_val="+str(np.min(val_x))]])           
    np.savetxt(f,title_cols, fmt="%s",delimiter="/")

    
    np.savetxt(f,(np.histogram(val_x, bins = train_bin_edges), np.histogram(train_x, bins = train_bin_edges)), fmt="%s",delimiter="\n")
    if options.test_ext == 'y': 
        #whole_internal_hist,whole_internal_bin_edges = np.histogram(whole_internal_set) 
        title_cols = np.array([["max_ext="+ str(np.max(v_data)),"min_ext="+str(np.min(v_data))]])           
        np.savetxt(f,title_cols, fmt="%s",delimiter="/")
        #np.savetxt(f,(np.histogram(v_data, bins = whole_internal_bin_edges)), fmt="%s",delimiter="\n")
        np.savetxt(f,(np.histogram(v_data, bins = train_bin_edges)), fmt="%s",delimiter="\n")
    f.close() 
    return train_x,val_x,v_data

def run_kfold_deepmg(options,args):
    '''
    run k fold cross validation with n times
    '''
    if options.type_run not in ['vis','visual']:  
        import tensorflow as tf
        #import models_def #models definitions
        from deepmg import models_def
        import keras as kr
        
        from keras.applications.resnet50 import ResNet50
        from keras.applications.vgg16 import VGG16
        from keras.models import Model
        from keras.applications.resnet50 import preprocess_input
        from keras.utils import np_utils
        from keras import backend as optimizers
        from keras.models import Sequential
        from keras.layers import Activation, Dropout, Flatten, Dense, InputLayer, Conv2D, MaxPooling2D,Input
        
    #declare varibables for checking points
    time_text = str(strftime("%Y%m%d_%H%M%S", gmtime())) 
    start_time = time.time() # check point the beginning
    mid_time = [] # check point middle time to measure distance between runs    
    input_shape = (options.dim_img, options.dim_img, options.channel) #input shape
      
    # ===================== STEP 1: NAMING folders and file pattern for outputs ===================
    # =============================================================================================
    
    path_write_parentfolder_img = []
    #print('test1eeeee')
    path_write_parentfolder_img, res_dir, pre_name = naming_folder(options)
    #print('test2eeeee')
    #get the pattern name for files log1,2,3
    prefix = name_log_final(options, n_time_text= str(time_text))
    ##print  ('prefix prefix prefix prefixprefix')
    ##print  (prefix
    
    name2_para = name_log_final(options, n_time_text= '')
    
    ##print  (prefix    )
    if options.debug == 'y':
        print  (prefix )
    #file1: contained general results, each fold; file2: accuracy,loss each run (finish a 10-cv), file3: Auc
    #folder details: acc,loss of each epoch, folder models: contain file of model (.json)
       
    prefix_file_log = res_dir + "/" + prefix + "file"
    prefix_details = res_dir + "/details/" + prefix + "details_"
    prefix_models = res_dir + "/models/" + prefix + "model_"

    # !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=
    # !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!= END : NAMING folders and file pattern for outputs >!=!=!=!=!=!=


    # ===================== STEP 2: read orginal data from excel ===================
    # ==============================================================================
    
    data, labels, name_features = read_data(options,get_names_of_features = 'y')   
    #print  ('length feature ' +  str(len(name_features)))
    v_data_ori = []
    v_labels = []
    if options.test_ext == 'y': #if use external validation set
        v_data_ori, v_labels = read_data(options,suffix='z',get_names_of_features = 'n', mode = 'external')
    
    if options.del0 == 'y': # Delete column if it is All zeros (0)
        #print  (data.shape)
        #print  ('delete columns which have all = 0')
        #data[:, (data != 0).any(axis=0)]
        data = data[:, (data != 0).any(axis=0)]
        #print  (data.shape  )   
        if options.test_ext == 'y':
            v_data_ori = v_data_ori[:,(v_data_ori != 0).any(axis=0)]
            ##print  (v_data_ori.shape[1])

    if options.test_ext == 'y': #if use external validation set, check #features of internal and external       
        if v_data_ori.shape[1] != data.shape[1]:
            print  ('external and internal sets do not have the same of features, ('+ utils.textcolor_display(str(v_data_ori.shape[1]) + ' != '+str(data.shape[1]))+")")
            print  (utils.textcolor_display( 'Please check option ' + utils.textcolor_display('--del0') + ' or your data') )
            exit()

            
    # !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=
    # !=!=!=!=!=!=!=!=!=!=!=!=!=!= end of read orginal data from excel !=!=!=!=!=!=!=
   
    #building cordinates for fill-up once
    # (a square of len_square*len_square containing all values of data)
    cordi_x = []
    cordi_y = []

    if options.fig_size == 0: #if fig_size <=0, use the smallest size which fit the data, generating images of (len_square x len_square)
        fig_v_size =  int(math.ceil(math.sqrt(data.shape[1])))    
    else:
        fig_v_size = options.fig_size

    if options.type_emb in [ 'fill','fills','zfill','zfills'] and options.algo_redu in  ['','none']: 
        #if use dimension reduction (algo_redu!=""), only applying after reducing
        if options.type_emb in ['zfill','zfills']:
            cordi_x, cordi_y = vis_data.coordinates_z_fillup(data.shape[1])    
        else:
            cordi_x, cordi_y = vis_data.coordinates_fillup(data.shape[1])    
        #create full set of images for fill-up using predefined-bins
        if options.type_emb in ['fill','zfill']: 
                 
            data = create_fillup_folder (options,path_write_parentfolder_img,cordi_x,cordi_y,
                    data,labels, prefix_img = 'fill')
            if options.test_ext == 'y': #if use external validation set           
                v_data_ori = create_fillup_folder (options,path_write_parentfolder_img,cordi_x,cordi_y,
                    v_data_ori,v_labels, prefix_img = 'testval')            

    
    #if use external validations, CREATING LOG FILE 4
    if options.test_ext == 'y':        
        
        #title_cols = np.array([["val_acc","val_auc","val_loss","val_mcc","tn", "fp", "fn", "tp",
        #    "index_fold","seed"]])  
        title_cols = np.array([["train_acc","val_acc","train_auc","val_auc","train_loss","val_loss","val_mcc","tn", "fp", "fn", "tp",
            "time","ep_stopped", "index_fold","seed"]])
        #in external evaluation:
        # # train ---> val (in internal set)
        # # val --> test (in external set)
        f=open(prefix_file_log+"_ext.txt",'a')
        np.savetxt(f,title_cols, fmt="%s",delimiter="\t")
        f.close()

    #save arg/options/configurations used in the experiment to LOG file1
    f = open(prefix_file_log+"_sum.txt",'a')
    np.savetxt(f,(options, args), fmt="%s", delimiter="\t")
    f.close()

    if options.save_para == 'y' or options.type_run in ['config']: #IF save para to config file to rerun    
        configfile_w = ''    
        if options. path_config_w == "": #if did not specify the path, so use the same folder with the path where the results saved
            #configfile_w = res_dir + "/" + pre_name + "_" + name2_para + ".cfg"
            configfile_w = res_dir + "/" + prefix + ".cfg"
        else: #if user want to specify the path
            #configfile_w = options. path_config_w + "/" + pre_name + "_" + name2_para + ".cfg"
            configfile_w = options.path_config_w + "/" + prefix + ".cfg"

        #if os.path.isfile(configfile_w):
        #    #print  ('the config file exists!!! Please remove it, and try again!')
        #    exit()
        ##print  (options)
        #else:
        write_para(options,configfile_w)  
        #print  ('Config file was saved to ' + utils.textcolor_display(configfile_w, type_mes = 'inf')  )

        if options.type_run in ['config']:
            exit()
       
    
    if options.save_avg_run in ['y']:
        #save title (column name) for log2: acc,loss each run with mean(k-fold-cv)
           
        #if options.save_l in ['y']:
        title_cols = np.array([["seed","ep","train_acc_start","train_acc_end","val_acc_begin","val_acc_end","train_loss_start","train_loss_end","val_loss_start","val_loss_end""total_time","time_distant"]])  
        #else:
        #    title_cols = np.array([["seed","ep","train_start","train_fin","test_start","test_fin","total_t","t_distant"]])  
        
        f=open(prefix_file_log+"_mean_acc.txt",'a')
        np.savetxt(f,title_cols, fmt="%s",delimiter="\t")
        f.close()
        
        #save title (column name) for log3: acc,auc, confusion matrix each run with mean(k-fold-cv)
        title_cols = np.array([["se","ep","tr_tn","tr_fn","tr_fp","tr_tp","va_tn","va_fn","va_fp","va_tp","tr_acc",
            "va_acc","tr_auc","va_auc"]])  
        f=open(prefix_file_log+"_mean_auc.txt",'a')
        np.savetxt(f,title_cols, fmt="%s",delimiter="\t")
        f.close()

    #save title (column name) for log of eachfolds: acc,loss,auc, mcc (k-fold-cv)
    if options.save_folds == 'y':

        #title_cols = np.array([["t_acc","v_acc","t_auc","v_auc",'v_mmc',"t_los","v_los","tn", "fp", "fn", "tp","time","ep_stopped"]]) 
        title_cols = np.array([["train_acc","val_acc","train_auc","val_auc","train_loss","val_loss","val_mmc","tn", "fp", "fn", "tp",
            "time","ep_stopped", "index_fold","seed"]])
        f=open(prefix_file_log+"_eachfold.txt",'a')
        np.savetxt(f,title_cols, fmt="%s",delimiter="\t")
        f.close()
    #print('test1e')
    ## initialize variables for global acc, auc
    temp_all_acc_loss=[]
    temp_all_auc_confusion_matrix=[]    

    train_acc_all =[] #stores the train accuracies of all folds, all runs
    train_auc_all =[] #stores the train auc of all folds, all runs
    #train_mmc_all =[] #stores the train auc of all folds, all runs
    val_mmc_all =[] #stores the validation accuracies of all folds, all runs
    val_acc_all =[] #stores the validation accuracies of all folds, all runs
    val_auc_all =[] #stores the validation auc of all folds, all runs
    
    #best_mmc_allfolds_run = -1
    #best_mmc_allfolds_run_get = -1
    #best_acc_allfolds_run = 0
    #best_auc_allfolds_run = 0

    # ===================== BEGIN: run each run ================================================
    # ==========================================================================================
    for seed_value in range(options.seed_value_begin,options.seed_value_begin+options.run_time) :
        #print  ("We're on seed %d" % (seed_value)    )
        np.random.seed(seed_value) # for reproducibility    
        if options.type_run not in ['vis','visual']:   
            #tf.set_random_seed(seed_value)
            try: 
                tf.set_random_seed(seed_value)
            except Exception as exception:
                tf.random.set_seed(seed_value)
        #rn.seed(seed_value)       

        #ini variables for scores in a seed
        #TRAIN: acc (at beginning + finish), auc, loss, tn, tp, fp, fn
        train_acc_1se=[]
        train_auc_1se=[]
        train_loss_1se=[]
        train_acc_1se_begin=[]
        train_loss_1se_begin=[]

        train_tn_1se=[]
        train_tp_1se=[]
        train_fp_1se=[]
        train_fn_1se=[]

        early_stop_1se=[]
       
        #VALIDATION: acc (at beginning + finish), auc, loss, tn, tp, fp, fn
        val_acc_1se=[]
        val_auc_1se=[]
        val_loss_1se=[]
        val_acc_1se_begin=[]
        val_loss_1se_begin=[]
        val_pre_1se=[]
        val_recall_1se=[]
        val_f1score_1se=[]
        val_mmc_1se=[]

        val_tn_1se=[]
        val_tp_1se=[]
        val_fp_1se=[]
        val_fn_1se=[]       

        #set the time distance between 2 runs, the time since the beginning
        distant_t=0
        total_t=0     
        v_val_auc_score = -1
        #begin each fold
        skf=StratifiedKFold(n_splits=options.n_folds, random_state=seed_value, shuffle= True)
        for index, (train_indices,val_indices) in enumerate(skf.split(data, labels)):
            if options.type_run in ['vis','visual']:
                print  ("Creating images on  " + str(seed_value) + ' fold' + str(index+1) + "/"+str(options.n_folds)+"..."     )  
            else:
                print  ("Training on  " + str(seed_value) + ' fold' + str(index+1) + "/"+str(options.n_folds)+"..."       )
            if options.debug == 'y':
                print(train_indices)
                print(val_indices)         

            #transfer data to train/val sets
            train_x = []
            train_y = []
            val_x = []
            val_y = []   
            v_data = []
            train_x, val_x = data[train_indices], data[val_indices]
            train_y, val_y = labels[train_indices], labels[val_indices]
            if options.test_ext == 'y':  
                v_data =  v_data_ori
            #print  (train_x.shape)
            ##### reduce dimension 
            if not (options.algo_redu in  ['','none']): #if use reducing dimension
                train_x, val_x, v_data = reduce_dim (options,prefix_file_log, seed_value, index, train_x, val_x, v_data)
                #apply cordi to new size
                #when appyling to reduce dimension, no more data which still = 0, so should not use type_dat='pr'
                if options.type_emb in [ 'fills','zfills']:                       
                    
                    if options.type_emb in ['zfills']:                        
                        cordi_x, cordi_y = vis_data.coordinates_z_fillup(options.new_dim)     
                    else:                        
                        cordi_x, cordi_y = vis_data.coordinates_fillup(options.new_dim)               
           
            #print(train_x.shape)
            #print(val_x.shape)

            
            train_x_original = train_x  

            ##### scale mode, use for creating images/bins
            if options.scale_mode in  ['none','']:
                print  ('do not use scaler')
            elif (options.scale_mode in ['mms','none','','qtf','qtnf','ln','ptf']) or options.scale_mode[0:3] == 'log' :
                train_x,val_x,v_data = scaler_section (options,prefix_file_log,seed_value,index,train_x,val_x,v_data)              
            else:
                print  ( utils.textcolor_display('this scaler (' + str(options.scale_mode) + ') is not supported now! Please try again!!'))
                exit()

            #select type_emb in ['raw','bin','fills','tsne',...]
            if options.type_emb == "raw" or options.type_emb == "bin" : #reshape raw data into a sequence for ltsm model. 
                
                if options.type_emb == "bin" : 
                    if options.auto_v  == 'y': #if auto, adjust automatically min_v and max_v for binning
                        #options.min_v = np.min(train_x)
                        #options.max_v = np.max(train_x)
                        min_v_temp = np.min(train_x)
                        max_v_temp = np.max(train_x)
                    else:
                        min_v_temp = options.min_v
                        max_v_temp = options.max_v
                    
                    temp_train_x=[]
                    temp_val_x=[]
                    temp_v_data=[]

                    #build bins and binning
                    bins_breaks = vis_data.build_breaks (data=train_x, type_bin=options.type_bin, num_bin=options.num_bin, max_v= max_v_temp, min_v = min_v_temp)

                    for i in range(0, len(train_x)):   
                        temp_train_x.append( [vis_data.convert_bin(value=y, min_v = min_v_temp, num_bin = options.num_bin, bins_break = bins_breaks) for y in train_x[i] ])
                    train_x = np.stack(temp_train_x)
                    for i in range(0, len(val_x)):   
                        temp_val_x.append( [vis_data.convert_bin(value=y, min_v = min_v_temp, num_bin = options.num_bin, bins_break = bins_breaks) for y in val_x[i] ])
                    val_x = np.stack(temp_val_x)

                    if options.test_ext == 'y':     
                        for i in range(0, len(v_data)):   
                            temp_v_data.append( [vis_data.convert_bin(value=y,  min_v = min_v_temp,  num_bin = options.num_bin, bins_break = bins_breaks) for y in v_data[i] ])
                        v_data = np.stack(temp_v_data)

                #print  ('use data...' + str(options.type_emb ))
                if options.model in ['model_con1d_baseline' , 'model_cnn1d']:
                    input_shape=((train_x.shape[1]),1)
                    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1],1))
                    val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1],1))
                    if options.test_ext == 'y':     
                        v_data = np.reshape(v_data, (v_data.shape[0], v_data.shape[1],1))

                elif options.model in [ 'svm_model','dtc_model','rf_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                    print('shape==')
                    print((train_x.shape[0], train_x.shape[1]))
                    input_shape=((train_x.shape[1]))
                    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))
                    val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1]))
                    if options.test_ext == 'y':     
                        v_data = np.reshape(v_data, (v_data.shape[0], v_data.shape[1]))
                    #print(train_x)
                else:
                    input_shape=(1,(train_x.shape[1]))
                    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
                    val_x = np.reshape(val_x, (val_x.shape[0], 1, val_x.shape[1]))      
                    if options.test_ext == 'y':     
                        v_data = np.reshape(v_data, (v_data.shape[0], 1, v_data.shape[1])) 
               
                #print('input_shape='+str(input_shape))
                ##print  ('train_x:=' + str(train_x.shape)        )      
               
                    
            elif not (options.type_emb in ["fill",'zfill']): #embedding type, new images after each k-fold

                #save information on bins
                f=open(prefix_file_log+"_sum.txt",'a')          
                if options.auto_v  == 'y': #if auto, adjust automatically min_v and max_v for binning
                    #options.min_v = np.min(train_x)
                    #options.max_v = np.max(train_x)
                    min_v_temp = np.min(train_x)
                    max_v_temp = np.max(train_x)
                else:
                    min_v_temp = options.min_v
                    max_v_temp = options.max_v
                    
                
                title_cols = np.array([["bins for classification"]])           
                np.savetxt(f,title_cols, fmt="%s",delimiter="/")               
                #save information on histogram of bins
                w_interval = float(max_v_temp - min_v_temp) / options.num_bin
                binv0 = []
                for ik in range(0,options.num_bin+1):
                    binv0.append(min_v_temp + w_interval*ik )

                np.savetxt(f,(np.histogram(val_x, bins = binv0), np.histogram(train_x, bins = binv0)), fmt="%s",delimiter="\n")
                f.close()

               
                # ++++++++++++ start embedding ++++++++++++

                train_x, val_x, v_data = visual_section (options = options,  train_x=train_x,val_x=val_x, train_y=train_y, val_y=val_y, v_data=v_data, 
                    v_labels=v_labels,train_x_original=train_x_original,
                    prefix_file_log=prefix_file_log, path_write_parentfolder_img=path_write_parentfolder_img, 
                    cordi_x=cordi_x,cordi_y=cordi_y,train_test_whole = 'n',seed_value=seed_value, index=index)
                ##print  (train_x)
                ##print  (train_x.shape[1],train_x.shape[2])

                
                #!=!=!=!=!=!= END of read images for train/val sets !=!=!=!=!=!=
                # !=!=!=!=!=!= END of embedding !=!=!=!=!=!=
               
            if options.debug == 'y':
                print(train_x.shape)
                print(val_x.shape)
                print(train_y)
                print(val_y)  
                print("size of data: whole data")
                print(data.shape)
                print("size of data: train_x")
                print(train_x.shape)
                print("size of data: val_x")
                print(val_x.shape)
                print("size of label: train_y")
                print(train_y.shape)
                print("size of label: val_y")
                print(val_y.shape)    

            f=open(prefix_file_log+"_sum.txt",'a')
            title_cols = np.array([["seed",seed_value," fold",index+1,"/",options.n_folds,"###############","seed",seed_value," fold",index+1,"/",options.n_folds,"###############","seed",seed_value," fold",index+1,"/",options.n_folds]])
            np.savetxt(f,title_cols, fmt="%s",delimiter="")
                        
            #save lables of train/test set to log
            if options.save_labels in ['y']:
                
                title_cols = np.array([["train_set"]])
                np.savetxt(f,title_cols, fmt="%s",delimiter="")        
                np.savetxt(f,[(train_y)], fmt="%s",delimiter=" ")
                title_cols = np.array([["validation_set"]])
                np.savetxt(f,title_cols, fmt="%s",delimiter="")    
                np.savetxt(f,[(val_y)], fmt="%s",delimiter=" ")
                
                f.close()   
                            
            if options.dim_img==-1 and options.type_emb != 'raw' and options.type_emb != 'bin': #if use real size of images
                print('shape===========================')
                print (train_x.shape)
                print (train_x.shape[0])
                
                input_shape = (train_x.shape[1],train_x.shape[2], options.channel)
            
            np.random.seed(seed_value) # for reproducibility   

            if options.type_run  in ['vis','visual']:  
                print  ('mode: VISUALIZATIONS, CREATING IMAGES, NOT LEARNING')
            else:
                print  ('mode: LEARNING')
                #tf.set_random_seed(seed_value)  
                try: 
                    tf.set_random_seed(seed_value)
                except Exception as exception:
                    tf.random.set_seed(seed_value)   
                #rn.seed(seed_value) 
                      
                if options.e_stop==-1: #=-1 : do not use earlystopping
                    options.e_stop = options.epoch

                if options.e_stop_consec=='consec': #use sefl-defined func
                    early_stopping = models_def.EarlyStopping_consecutively(monitor='val_loss', 
                        patience=options.e_stop, verbose=1, mode='auto')
                else: #use keras func
                    early_stopping = kr.callbacks.EarlyStopping(monitor='val_loss', 
                        patience=options.e_stop, verbose=1, mode='auto')

                if options.coeff == 0: #run tuning for coef
                    print  ( utils.textcolor_display ('Error! options.coeff == 0 only use for tuning coef....') )
                    exit()                                         
                        
                #+++++++++++ transform or rescale before fetch data into learning  +++++++++++
                #if options.num_classes >= 2:
                #    train_y=kr.utils.np_utils.to_categorical(train_y, num_classes = num_classes )
                #    val_y = kr.utils.np_utils.to_categorical(val_y, num_classes =num_classes)

                if options.debug == 'y':
                    print  ('max_before coeff=======')
                    print  (np.amax(train_x))
                train_x = train_x / float(options.coeff)
                val_x = val_x / float(options.coeff)   
                if options.test_ext == 'y':       
                    v_data = v_data/float(options.coeff)

                if options.num_classes >= 2:
                    if options.model in ['svm_model']:
                        from sklearn import preprocessing
                        encoder = preprocessing.LabelEncoder()
                        encoder.fit(train_y)
                        train_y = encoder.transform(train_y)

                        encoder.fit(val_y)
                        val_y = encoder.transform(val_y)
                        if options.test_ext == 'y':
                               
                            encoder.fit(v_labels)
                            v_labels = encoder.transform(v_labels)
                    else:
                        train_y=kr.utils.np_utils.to_categorical(train_y,  num_classes = options.num_classes )
                        val_y = kr.utils.np_utils.to_categorical(val_y ,  num_classes = options.num_classes )
                        if options.test_ext == 'y':
                            v_labels = kr.utils.np_utils.to_categorical(v_labels ,  num_classes = options.num_classes )
                
                if options.debug == 'y':
                    print  ('max_after coeff======='     )            
                    print  ('train='+str(np.amax(train_x)))
                    if options.test_ext == 'y':       
                        print  ('val_test='+str(np.amax(v_data)))
                #!=!=!=!=!=!= end of transform or rescale before fetch data into learning  !=!=!=!=!=!=


                #++++++++++++   selecting MODELS AND TRAINING, TESTING ++++++++++++
               
                
                if options.model in  ['resnet50', 'vgg16']:
                    input_shape = Input(shape=(train_x.shape[1],train_x.shape[2], options.channel)) 
                
                #if options.seed_ml == 'y':
                #    seed_algo = seed_value
                #else:
                #    seed_algo = None
                model = models_def.call_model (type_model=options.model, 
                        m_input_shape= input_shape, m_num_classes= options.num_classes, m_optimizer = options.optimizer,
                        m_learning_rate=options.learning_rate, m_learning_decay=options.learning_rate_decay, m_loss_func=options.loss_func,
                        ml_number_filters=options.numfilters,ml_numlayercnn_per_maxpool=options.numlayercnn_per_maxpool, ml_dropout_rate_fc=options.dropout_fc, 
                        mc_nummaxpool=options.nummaxpool, mc_poolsize= options.poolsize, mc_dropout_rate_cnn=options.dropout_cnn, 
                        mc_filtersize=options.filtersize, mc_padding = options.padding,
                        svm_c=options.svm_c, svm_kernel=options.svm_kernel,rf_n_estimators=options.rf_n_estimators,
                        rf_max_depth = options.rf_max_depth,
                        m_pretrained_file = options.pretrained_w_path, knn_n_neighbors=options.knn_n_neighbors,
                        rf_max_features= options.rf_max_features
                        #,                        rf_random_state = seed_algo
                        )
                
                        
                if options.visualize_model == 'y':  #visualize architecture of model  
                    #from keras_sequential_ascii import sequential_model_to_ascii_printout
                    #sequential_model_to_ascii_printout(model)    
                    from keras_sequential_ascii import keras2ascii
                    keras2ascii(model)                  

                np.random.seed(seed_value) # for reproducibility     
                #tf.set_random_seed(seed_value) 
                try: 
                    tf.set_random_seed(seed_value)
                except Exception as exception:
                    tf.random.set_seed(seed_value)  
                #rn.seed(seed_value)                  
                        
                if options.model in ['svm_model','dtc_model', 'rf_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                    #print  (train_x.shape      )     
                    #print  ('run classic learning algorithms'  )
                    
                    if np.isnan(train_x.any()):
                        print  ( utils.textcolor_display ('data contain NaN. Please check data again'))
                        exit()
                    elif np.isfinite(train_x.all()):
                        print  ('data all are finite ')
                    else: 
                        print  ( utils.textcolor_display ('data contain infitite value. Please check data again'))
                        exit()
                    print(train_x.shape)
                    print(train_y)

                    model.fit(train_x, train_y)

                    if options.model in ['rf_model','gbc_model','dtc_model']: #save important and scores of features in Random Forests                    
                        if  options.save_rf == 'y':
                            importances = model.feature_importances_
                            ##print  (importances)
                            #indices_imp = np.argsort(importances)
                            #indices_imp = np.argsort(importances)[::-1] #reserved order
                            f=open(prefix_details+"s"+str(seed_value)+"k"+str(index+1)+"_importance_fea.csv",'a')
                            np.savetxt(f,np.c_[(name_features,importances)], fmt="%s",delimiter=",")
                            f.close()
                else:
                    #print  ('run deep learning algorithms')
                    #print (train_x.shape)
                    history_callback=model.fit(train_x, train_y, 
                        epochs = options.epoch, 
                        batch_size=options.batch_size, verbose=1,
                        validation_data=(val_x, val_y), callbacks=[early_stopping],
                        shuffle=False)        # if shuffle=False could be reproducibility
                    #print  (history_callback)
            #!=!=!=!=!=!= end of selecting MODELS AND TRAINING, TESTING  !=!=!=!=!=!=

            #++++++++++++SAVE ACC, LOSS of each epoch++++++++++++     
           

            if options.type_run in ['vis','visual']: #only visual, not learning
                print  ('mode: VISUALIZATIONS, CREATING IMAGES, NOT LEARNING, skip writing logs.... each fold')

            else: #learning
                train_acc =[]
                train_loss = []
                val_acc = []
                val_loss = []

                if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
            
                    #ep_arr=range(1, 2, 1) #epoch    
                    ep_arr = []       
                    ep_arr.append(0)
                    ep_arr.append(1)

                    train_acc.append (-1)
                    train_loss.append (-1)
                    val_acc .append (-1)
                    val_loss .append (-1)


                    Y_pred = model.predict_proba(train_x)#, verbose=2)
                    #Y_pred = model.predict(train_x) #will check auc later
                    if options.debug == 'y':
                        print  (Y_pred)
                        print  (Y_pred[:,1])
                    ##print  (Y_pred.loc[:,0].values
                    ##print  (y_pred
                    if options.num_classes > 2:
                        train_auc_score =  -1
                        
                        #train_auc_score = multiclass_roc_auc_score(train_y, Y_pred[:,1])
                        #print(Y_pred)
                        #print(train_y)
                        #factor = pd.factorize(train_y)
                        #reversefactor = dict(zip(range(options.num_classes),factor[0]))
                        #y_label_re = np.vectorize(reversefactor.get)(train_y)
                        #y_pred_re = np.vectorize(reversefactor.get)(Y_pred)
                        #train_acc.append ( accuracy_score(y_label_re, y_pred_re))
                        #train_loss.append ( log_loss(y_label_re, y_pred_re))
                        #print(Y_pred)
                        #print(confusion_matrix(train_y,Y_pred))
                        #from sklearn import preprocessing
                        #encoder = preprocessing.LabelEncoder()
                        #print(Y_pred)
                        #Y_pred_label = list(encoder.inverse_transform(Y_pred))
                        #print(Y_pred_label)
                        #print(Y_pred)
                        #print(train_y)
                        #print(np.argmax(Y_pred, axis=1) )  
                        #train_y = kr.utils.np_utils.to_categorical(train_y ,  num_classes = options.num_classes )
                        #print(train_y)
                        train_acc.append ( accuracy_score(train_y, np.argmax(Y_pred, axis=1)))
                        #print('argmax')
                        #print(np.argmax(Y_pred, axis=1) )  
                        #print('train_y')
                        #print(train_y)
                        #train_loss.append ( log_loss(train_y, np.argmax(Y_pred, axis=1)))
                        train_loss.append ( -1)
                        train_mmc = -2

                        Y_pred_val = model.predict_proba(val_x)
                        #val_acc.append ( accuracy_score(val_y, Y_pred_val[:,1].round()))
                        val_acc.append ( accuracy_score(val_y, np.argmax(Y_pred_val, axis=1)))
                        #val_loss.append ( -1)
                        print(val_y)
                        #print(np.max(Y_pred_val, axis=1) * options.num_classes)
                        print(np.argmax(Y_pred_val, axis=1))
                        #val_loss.append ( log_loss(val_y, np.max(Y_pred_val, axis=1) * options.num_classes))
                        #val_loss.append ( log_loss(val_y, np.argmax(Y_pred_val, axis=1)))
                        val_loss.append ( -1)
                    else:
                        train_auc_score = roc_auc_score(train_y, Y_pred[:,1])
                        train_acc.append ( accuracy_score(train_y, Y_pred[:,1].round()))
                        train_loss.append ( log_loss(train_y, Y_pred[:,1]))
                    
                        Y_pred_val = model.predict_proba(val_x)
                        #val_acc.append ( accuracy_score(val_y, Y_pred_val[:,1].round()))
                        val_acc.append ( accuracy_score(val_y, Y_pred_val[:,1].round()))
                        #val_loss.append ( -1)
                        val_loss.append ( log_loss(val_y, Y_pred_val[:,1]))

                        train_mmc = matthews_corrcoef(train_y,Y_pred[:,1].round())
                else:
                    try: 
                        #ep_arr=range(1, options.epoch+1, 1) #epoch
                        ep_arr=range(1, len(history_callback.history['acc'])+1, 1) #epoch
                        train_acc = history_callback.history['acc']  #acc
                        train_loss = history_callback.history['loss'] #loss
                        val_acc = history_callback.history['val_acc']  #acc
                        val_loss = history_callback.history['val_loss'] #loss
                    except:
                        ep_arr=range(1, len(history_callback.history['accuracy'])+1, 1) #epoch
                        train_acc = history_callback.history['accuracy']  #acc
                        train_loss = history_callback.history['loss'] #loss
                        val_acc = history_callback.history['val_accuracy']  #acc
                        val_loss = history_callback.history['val_loss'] #loss

                    Y_pred = model.predict(train_x, verbose=2)

                    if options.num_classes > 2:
                        train_auc_score = -1
                        #train_auc_score = multiclass_roc_auc_score(train_y, Y_pred)
                    else:
                        train_auc_score = roc_auc_score(train_y, Y_pred)
                    if options.num_classes > 2:
                        train_mmc = -2
                    else:
                        train_mmc = matthews_corrcoef(train_y,Y_pred.round())

                #store for every epoch
                title_cols = np.array(["ep","train_loss","train_acc","val_loss","val_acc"])  
                res=(ep_arr,train_loss,train_acc,val_loss,val_acc)
                res=np.transpose(res)
                combined_res=np.array(np.vstack((title_cols,res)))
                #combined_res=np.array((title_cols,res))

                #save to file, a file contains acc,loss of all epoch of this fold
                #print('prefix_details====')            
                #print(prefix_details)   
                #if options.save_optional in ['2','3','6','7'] :  
                if options.save_d in ['y'] :             
                    np.savetxt(prefix_details+"s"+str(seed_value)+"k"+str(index+1)+".txt", 
                        combined_res, fmt="%s",delimiter="\t")
                
                #!=!=!=!=!=!= END of SAVE ACC, LOSS of each epoch !=!=!=!=!=!=

                #++++++++++++ CONFUSION MATRIX and other scores ++++++++++++
                #train
                ep_stopped = len(train_acc)
                
                #print  ('epoch stopped= '+str(ep_stopped)
                early_stop_1se.append(ep_stopped)

                train_acc_1se.append (train_acc[ep_stopped-1])
                train_acc_1se_begin.append (train_acc[0] )
                train_loss_1se.append (train_loss[ep_stopped-1])
                train_loss_1se_begin.append (train_loss[0] )

              
                
                if options.num_classes >= 2:   
                    y_pred = np.argmax(Y_pred, axis=1)       
                    print('---')
                    print(y_pred)
                    print('------')
                    #print (np.argmax(train_y,axis=1))
                    if options.model in ['svm_model']:
                        cm = confusion_matrix(train_y,y_pred)
                        print(cm)
                    else:
                        cm = confusion_matrix(np.argmax(train_y,axis=1),y_pred)
                    if options.num_classes > 2: 
                        tn = -1
                        fp = -1
                        fn = -1
                        tp = -1
                    else:
                        tn, fp, fn, tp = cm.ravel()                
                    #print("confusion matrix training")
                    #print(cm.ravel())
                    
                else:
                    if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                        if options.debug == 'y':
                            print(Y_pred[:,1])
                        cm = confusion_matrix(train_y,Y_pred[:,1].round())
                        if options.num_classes > 2: 
                            tn = -1
                            fp = -1
                            fn = -1
                            tp = -1
                        else:
                            tn, fp, fn, tp = cm.ravel()
                    
                        #print("confusion matrix training")
                        #print(cm.ravel())
                        
                    else:
                        if options.debug == 'y':
                            print(Y_pred.round())
                        ##print  (train_y
                        cm = confusion_matrix(train_y,Y_pred.round())
                        if options.num_classes > 2: 
                            tn = -1
                            fp = -1
                            fn = -1
                            tp = -1
                        else:
                            tn, fp, fn, tp = cm.ravel()
                    
                        #print("confusion matrix training")
                        #print(cm.ravel())
                
                train_tn_1se.append (tn)
                train_fp_1se.append (fp)
                train_fn_1se.append (fn) 
                train_tp_1se.append (tp)
                train_auc_1se.append (train_auc_score)

                #if math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) == 0:
                #    train_mmc = 0
                #else:
                #    train_mmc = np.around(float(tp*tn - fp*fn) / float(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),decimals=3)
                #train_mmc = matthews_corrcoef(train_y,Y_pred)
                

                if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                    #Y_pred = model.predict(val_x)
                    Y_pred = model.predict_proba(val_x)#, verbose=2)
                    if options.debug == 'y':
                        print  ('scores from svm/rf')
                        print  (Y_pred)
                    
                    if options.num_classes > 2:
                        val_auc_score = -1
                        #val_auc_score = multiclass_roc_auc_score(val_y, Y_pred[:,1])
                    else:
                        val_auc_score = roc_auc_score(val_y, Y_pred[:,1])
                    val_acc.append ( accuracy_score(val_y, Y_pred[:,1].round()))
                    
                    #val_loss.append ( log_loss(val_y, Y_pred[:,1]))
                    val_loss.append ( -1)
                    if options.num_classes > 2:
                        val_mmc = -2
                    else:
                        val_mmc = matthews_corrcoef(val_y,Y_pred[:,1].round())
                else:
                    Y_pred = model.predict(val_x, verbose=2)
                    if options.debug == 'y':
                        print  ('scores from nn')
                        print  (Y_pred)
                    if options.num_classes > 2:
                        val_auc_score = -1
                        #val_auc_score = multiclass_roc_auc_score(val_y, Y_pred)
                    else:
                        val_auc_score=roc_auc_score(val_y, Y_pred)
                    if options.num_classes > 2:
                        val_mmc = -2
                    else: 
                        val_mmc = matthews_corrcoef(val_y,Y_pred.round())
                ##print(Y_pred)
                #print("score auc" +str(val_auc_score))          
                
                #store to var, global acc,auc  
                #test
                val_acc_1se.append (val_acc[ep_stopped-1])
                val_acc_1se_begin.append (val_acc[0] )
                val_loss_1se.append (val_loss[ep_stopped-1])
                val_loss_1se_begin.append (val_loss[0] )

                if options.num_classes>=2:     
                    if options.model in ['svm_model']:
                        #print('(--------)')
                        #print(val_y)
                        #print('(-----ypred---)')
                        #print(np.argmax(Y_pred,axis=1))
                        cm = confusion_matrix(val_y,np.argmax(Y_pred,axis=1))
                        
                    else:
                        y_pred = np.argmax(Y_pred, axis=1)    
                        cm = confusion_matrix(np.argmax(val_y,axis=1),y_pred)
                    
                    print(cm)
                    if options.num_classes > 2: 
                        tn = -1
                        fp = -1
                        fn = -1
                        tp = -1
                    else:
                        tn, fp, fn, tp = cm.ravel()
                    #print("confusion matrix val")
                    #print(cm.ravel())
                else:
                    if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                        if options.debug == 'y':
                            print(Y_pred[:,1])
                        cm = confusion_matrix(val_y,Y_pred[:,1].round())
                        if options.num_classes > 2: 
                            tn = -1
                            fp = -1
                            fn = -1
                            tp = -1
                        else:
                            tn, fp, fn, tp = cm.ravel()
                    
                        #print("confusion matrix val")
                        #print(cm.ravel())
                    else:
                        if options.debug == 'y':
                            print(Y_pred.round())
                        cm = confusion_matrix(val_y,Y_pred.round())
                        if options.num_classes > 2: 
                            tn = -1
                            fp = -1
                            fn = -1
                            tp = -1
                        else:
                            tn, fp, fn, tp = cm.ravel()
                        #print("confusion matrix val")
                        #print(cm.ravel())
                
                val_tn_1se.append (tn)
                val_fp_1se.append (fp)
                val_fn_1se.append (fn) 
                val_tp_1se.append (tp)
                val_auc_1se.append (val_auc_score)

               

                # Accuracy = TP+TN/TP+FP+FN+TN
                # Precision = TP/TP+FP
                # Recall = TP/TP+FN
                # F1 Score = 2 * (Recall * Precision) / (Recall + Precision)
                # TP*TN - FP*FN / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

                if tp+fp == 0:
                    val_pre = 0
                else:
                    val_pre = np.around( tp/float(tp+fp),decimals=3)

                if tp+fn == 0:
                    val_recall=0
                else:
                    val_recall = np.around(tp/float(tp+fn),decimals=3)
                
                if val_recall+val_pre == 0:
                    val_f1score = 0
                else:
                    val_f1score = np.around(2 * (val_recall*val_pre)/ float(val_recall+val_pre),decimals=3)

                #if math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) == 0:
                #    val_mmc = 0
                #else:
                #    val_mmc = np.around(float(tp*tn - fp*fn) / float(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),decimals=3)
                
                #val_mmc = matthews_corrcoef(val_y,Y_pred)

                #!=!=!=!=!=!= END of CONFUSION MATRIX and other scores  !=!=!=!=!=!=

                val_pre_1se.append (val_pre)
                val_recall_1se.append (val_recall)
                val_f1score_1se.append (val_f1score)
                val_mmc_1se.append (val_mmc)

                f=open(prefix_file_log+"_sum.txt",'a')   
                t_checked=time.time()           
                # title_cols = np.array([["seed",seed_value," fold",index+1,"/",options.n_folds]])
                # np.savetxt(f,title_cols, fmt="%s",delimiter="")
                title_cols = np.array([["after training"]])           
                np.savetxt(f,title_cols, fmt="%s",delimiter="\t")
                title_cols = np.array([["t_acc","v_acc","t_auc","v_auc","t_los","v_los",'v_mmc',"tn", "fp", "fn", "tp","time","ep_stopped"]])           
                np.savetxt(f,title_cols, fmt="%s",delimiter="--")
                np.savetxt(f,np.around(np.c_[ 
                    ( train_acc[ep_stopped-1],val_acc[ep_stopped-1], 
                      train_auc_score,val_auc_score,                    
                      train_loss[ep_stopped-1] , val_loss[ep_stopped-1],val_mmc,tn, fp, fn, tp, (t_checked - start_time),ep_stopped)
                    ],decimals=3), fmt="%s",delimiter="++")
                f.close()

                if options.save_folds == 'y':
                    
                    #title_cols = np.array([["train_acc","val_acc","train_auc","val_auc","val_mmc","train_loss","val_loss","val_mcc","tn", "fp", "fn", "tp",
                    #    "time","ep_stopped", "index_fold","seed"]])
                    f=open(prefix_file_log+"_eachfold.txt",'a')
                    np.savetxt(f,np.around(np.c_[( train_acc[ep_stopped-1], val_acc[ep_stopped-1], train_auc_score, val_auc_score,                            
                            train_loss[ep_stopped-1], val_loss[ep_stopped-1], val_mmc,tn, fp, fn, tp, (t_checked - start_time),ep_stopped, 
                            index, seed_value
                        )], decimals=8), fmt="%s",delimiter="\t")
                    f.close() 

                train_acc_all.append(train_acc[ep_stopped-1])
                train_auc_all.append(train_auc_score)
                val_acc_all.append(val_acc[ep_stopped-1])
                val_auc_all.append(val_auc_score)
                val_mmc_all.append(val_mmc)
            
                #use external set
                if options.test_ext == 'y':                      
                    #auc score, acc, mmc
                    if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                        Y_pred_v = model.predict_proba(v_data)#, verbose=2)               
                        if options.num_classes > 2:
                            val_auc_score = -1
                            #val_auc_score = multiclass_roc_auc_score(v_labels, Y_pred[:,1])
                        else:
                            val_auc_score = roc_auc_score(v_labels, Y_pred_v[:,1])
                        v_val_loss_score = log_loss (v_labels, Y_pred_v[:,1])
                        v_val_acc_score = accuracy_score(v_labels, Y_pred_v[:,1].round())  
                        if options.num_classes > 2:
                            v_val_mmc_score = -2
                        else:                 
                            v_val_mmc_score = matthews_corrcoef(v_labels, Y_pred_v[:,1].round())
                        
                    else:                                              
                        Y_pred_v = model.predict(v_data)      
                        if options.num_classes > 2:
                            v_val_auc_score = -1
                            #v_val_auc_score = multiclass_roc_auc_score(v_labels, Y_pred_v)
                        else:                  
                            v_val_auc_score = roc_auc_score(v_labels, Y_pred_v)
                        v_val_loss_score = log_loss(v_labels, Y_pred_v)
                        v_val_acc_score = accuracy_score(v_labels, Y_pred_v.round())  
                        if options.num_classes > 2:
                            v_val_mmc_score = -2
                        else:                       
                            v_val_mmc_score = matthews_corrcoef(v_labels, Y_pred_v.round())
                   

                    if options.debug == 'y':
                        print  (Y_pred)
                        
                    #confusion matrix
                    if options.num_classes>=2:     
                        y_pred = np.argmax(Y_pred_v, axis=1)    
                        cm = confusion_matrix(np.argmax(v_labels,axis=1),y_pred)
                        if options.num_classes > 2: 
                            tn = -1
                            fp = -1
                            fn = -1
                            tp = -1
                        else:
                            tn, fp, fn, tp = cm.ravel()
                        #print("confusion matrix external")
                        #print(cm.ravel())
                    else:
                        if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                            if options.debug == 'y':
                                print(Y_pred_v[:,1])
                            cm = confusion_matrix(v_labels,Y_pred_v[:,1].round())
                            if options.num_classes > 2: 
                                tn = -1
                                fp = -1
                                fn = -1
                                tp = -1
                            else:
                                tn, fp, fn, tp = cm.ravel()
                        
                            #print("confusion matrix external")
                            #print(cm.ravel())
                        else:
                            if options.debug == 'y':
                                print(Y_pred_v.round())
                            cm = confusion_matrix(v_labels,Y_pred_v.round())
                            if options.num_classes > 2: 
                                tn = -1
                                fp = -1
                                fn = -1
                                tp = -1
                            else:
                                tn, fp, fn, tp = cm.ravel()
                            #print("confusion matrix external")
                            #print(cm.ravel())
                    #title_cols = np.array([["train_acc","val_acc","train_auc","val_auc","train_loss","val_loss","val_mcc","tn", "fp", "fn", "tp",
                    #    "time","ep_stopped", "index_fold","seed"]])
                    #in external evaluation:
                    # # train ---> val (in internal set)
                    # # val --> test (in external set)
                    f=open(prefix_file_log+"_ext.txt",'a')   
                   
                    np.savetxt(f,np.c_[(
                        val_acc[ep_stopped-1], v_val_acc_score,
                        val_auc_score, v_val_auc_score,
                        val_loss[ep_stopped-1], v_val_loss_score, 
                        v_val_mmc_score ,tn, fp, fn, tp,-1,-1, index+1, seed_value)], fmt="%s",delimiter="\t")
                    f.close() 

                    #if (best_mmc_allfolds_run < val_mmc) or (best_mmc_allfolds_run == val_mmc and v_val_auc_score > best_auc_allfolds_run) or (best_mmc_allfolds_run == val_mmc and v_val_auc_score == best_auc_allfolds_run and v_val_acc_score > best_acc_allfolds_run):
                    #    best_mmc_allfolds_run = val_mmc
                    #    best_mmc_allfolds_run_get = v_val_mmc_score
                    #    best_acc_allfolds_run = v_val_acc_score
                    #    best_auc_allfolds_run = v_val_auc_score

            
                #save model file         
                
                if options.save_w == 'y' : #save weights        
                    #https://scikit-learn.org/stable/modules/model_persistence.html
                    if options.model in [ 'svm_model','rf_model','dtc_model','gbc_model','knn_model']:
                        import pickle
                        filename = prefix_models+"s"+str(seed_value)+"k"+str(index+1)+"ml.sav"
                        pickle.dump(model, open(filename, 'wb'))

                    else:
                        model.save_weights(prefix_models+"s"+str(seed_value)+"k"+str(index+1)+".h5")                    
                        model_json = model.to_json()
                        with open(prefix_models+"s"+str(seed_value)+"k"+str(index+1)+".json", "w") as json_file:
                            json_file.write(model_json)
                    #print("Saved model to disk")
        
            # !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!= finish one fold  !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=
            
        
        if options.type_run in ['vis','visual']: #only visual, not learning
            print  ('mode: VISUALIZATIONS, CREATING IMAGES, NOT LEARNING, skip writing logs.... each run time')

        else: #learning    
            #check point after finishing one running time
            mid_time.append(time.time())   
            if seed_value == options.seed_value_begin:              
                total_t=mid_time[seed_value - options.seed_value_begin ]-start_time
                distant_t=total_t
            else:
                total_t=mid_time[seed_value- options.seed_value_begin ]-start_time
                distant_t=mid_time[seed_value - options.seed_value_begin]-mid_time[seed_value-options.seed_value_begin-1 ]           

            res=(seed_value,np.mean(early_stop_1se),
                np.mean(train_acc_1se_begin),np.mean(train_acc_1se),
                np.mean(val_acc_1se_begin),np.mean(val_acc_1se),
                np.mean(train_loss_1se_begin),np.mean(train_loss_1se),
                np.mean(val_loss_1se_begin),np.mean(val_loss_1se),
                total_t,distant_t)
            temp_all_acc_loss.append(res)       
        
            if options.save_avg_run in ['y']:
                f=open(prefix_file_log+"_mean_acc.txt",'a')        
                #if options.save_l in ['y'] :
                np.savetxt(f,np.c_[(seed_value,np.mean(early_stop_1se),
                        np.mean(train_acc_1se_begin),np.mean(train_acc_1se),
                        np.mean(val_acc_1se_begin),np.mean(val_acc_1se),
                        np.mean(train_loss_1se_begin),np.mean(train_loss_1se),
                        np.mean(val_loss_1se_begin),np.mean(val_loss_1se),
                        total_t,distant_t)], fmt="%s",delimiter="\t")
                #else:
                #    np.savetxt(f,np.c_[(seed_value,np.mean(early_stop_1se),
                #        np.mean(train_acc_1se_begin),np.mean(train_acc_1se),
                #        np.mean(val_acc_1se_begin),np.mean(val_acc_1se),
                    #  train_loss_1se_begin.mean(),train_loss_1se.mean(),
                    #  val_loss_1se_begin.mean(),val_loss_1se.mean(),
                #        total_t,distant_t)],fmt="%s",delimiter="\t")
                f.close()

                #confusion and auc
            res=(seed_value,np.mean(early_stop_1se),
                        np.mean(train_tn_1se),np.mean(train_fn_1se),
                        np.mean(train_fp_1se),np.mean(train_tp_1se),
                        np.mean(val_tn_1se),np.mean(val_fn_1se),
                        np.mean(val_fp_1se),np.mean(val_tp_1se),
                        (float((np.mean(train_tp_1se)+np.mean(train_tn_1se))/float(np.mean(train_tn_1se)+np.mean(train_fn_1se)+np.mean(train_fp_1se)+np.mean(train_tp_1se)))), #accuracy
                        (float((np.mean(val_tp_1se)+np.mean(val_tn_1se))/float(np.mean(val_tn_1se)+np.mean(val_fn_1se)+np.mean(val_fp_1se)+np.mean(val_tp_1se)))), 
                        float(np.mean(train_auc_1se)) , #auc
                        float(np.mean(val_auc_1se)) ,
                        float(np.mean(val_pre_1se)),
                        float(np.mean(val_recall_1se)),
                        float(np.mean(val_f1score_1se)),
                        float(np.mean(val_mmc_1se)))
            temp_all_auc_confusion_matrix.append(res)    

            if options.save_avg_run in ['y']:
                f=open(prefix_file_log+"_mean_auc.txt",'a')
                np.savetxt(f,np.c_[(seed_value,np.mean(early_stop_1se),
                        np.mean(train_tn_1se),np.mean(train_fn_1se),
                        np.mean(train_fp_1se),np.mean(train_tp_1se),
                        np.mean(val_tn_1se),np.mean(val_fn_1se),
                        np.mean(val_fp_1se),np.mean(val_tp_1se),
                        (float((np.mean(train_tp_1se)+np.mean(train_tn_1se))/float(np.mean(train_tn_1se)+np.mean(train_fn_1se)+np.mean(train_fp_1se)+np.mean(train_tp_1se)))), #accuracy
                        (float((np.mean(val_tp_1se)+np.mean(val_tn_1se))/float(np.mean(val_tn_1se)+np.mean(val_fn_1se)+np.mean(val_fp_1se)+np.mean(val_tp_1se)))), 
                        float(np.mean(train_auc_1se)) , #auc
                        float(np.mean(val_auc_1se)))],fmt="%s",delimiter="\t")
                f.close()
        # !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!= END of all folds of a running time !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=
    
    
    if options.type_run in ['vis','visual']: #only visual, not learning
        print  ('mode: VISUALIZATIONS, CREATING IMAGES, NOT LEARNING, skip writing logs.... skip collecting results')
        print  ('images were created successfully at ' + path_write)

    else: #learning  
        #update to the end of file the mean/sd of all results (acc,loss,...): file2
       
        acc_loss_mean=np.around(np.mean(temp_all_acc_loss, axis=0),decimals=3)
        acc_loss_std=np.around(np.std(temp_all_acc_loss, axis=0),decimals=3)

        if options.save_avg_run in ['y']:
            f=open(prefix_file_log+"_mean_acc.txt",'a')
            np.savetxt(f,acc_loss_mean, fmt="%s",newline="\t")
            np.savetxt(f,acc_loss_std, fmt="%s",newline="\t",header="\n")
            f.close() 
       

        #update to the end of file the mean/sd of all results (auc, tn, tp,...): file3
        
        auc_mean=np.around(np.mean(temp_all_auc_confusion_matrix, axis=0),decimals=3)
        auc_std=np.around(np.std(temp_all_auc_confusion_matrix, axis=0),decimals=3)
        
        if options.save_avg_run in ['y']:
            f=open(prefix_file_log+"_mean_auc.txt",'a')
            np.savetxt(f,auc_mean, fmt="%s",newline="\t")
            np.savetxt(f,auc_std, fmt="%s",newline="\t",header="\n")
            f.close() 

        finish_time=time.time()
        #save final results to the end of file1
        
        
        f=open(prefix_file_log+"_sum.txt",'a')
        title_cols = np.array([['time','tr_ac',"va_ac","sd","va_au","sd","tn","fn","fp","tp","preci","sd","recall","sd","f1","sd","mmc","sd","epst"]])
        np.savetxt(f,title_cols, fmt="%s",delimiter="\t")   
        TN=auc_mean[6]
        FN=auc_mean[7]
        FP=auc_mean[8]
        TP=auc_mean[9]
        precision=auc_mean[14]
        recall=auc_mean[15]
        f1_score=auc_mean[16]
        mmc=auc_mean[17]

        np.savetxt(f,np.c_[ (np.around( finish_time-start_time,decimals=1),acc_loss_mean[3],acc_loss_mean[5],acc_loss_std[5],auc_mean[13],auc_std[13], 
            TN, FN, FP, TP,precision,auc_std[14],recall,auc_std[15],f1_score,auc_std[16],mmc,auc_std[17],auc_mean[1])],
            fmt="%s",delimiter="\t")
        

       
        title_cols = np.array([['tr_ac_a','sd_ac',"va_ac_a","sd_ac",'tr_au_a','sd_au',"va_au_a","sd_au",'va_mc_a','sd_mc']])
        np.savetxt(f,title_cols, fmt="%s",delimiter="\t")   
        np.savetxt(f,np.c_[ (
            np.mean(train_acc_all, axis=0),
            np.std(train_acc_all, axis=0),
            np.mean(val_acc_all, axis=0),
            np.std(val_acc_all, axis=0),
            #len(train_acc_all), #in order to check #folds * #run
            #len(val_acc_all),
            np.mean(train_auc_all, axis=0),
            np.std(train_auc_all, axis=0),
            np.mean(val_auc_all, axis=0),
            np.std(val_auc_all, axis=0),
            np.mean(val_mmc_all, axis=0),
            np.std(val_mmc_all, axis=0)
            #len(train_auc_all),
            #len(val_auc_all)
            )] , fmt="%s",delimiter="\t")
        
        f.close()   
        #append "ok" as marked "done!"          
        
        if options.cudaid > -3:    
        #if options.cudaid not in  ["-1",'']:
            #os.rename(res_dir + "/" + prefix+"file_sum.txt", res_dir + "/" + prefix+"gpu"+ str(options.cudaid) + "file_sum_"+str(options.suff_fini)+".txt")  
            os.rename(res_dir + "/" + prefix+"file_sum.txt", res_dir + "/" + prefix+"gpu_file_sum_"+str(options.suff_fini)+".txt")  
        else:
            os.rename(prefix_file_log+"_sum.txt", prefix_file_log+"_sum_"+str(options.suff_fini)+".txt")  
        
        print ('mean_val_acc=' + str(acc_loss_mean[5]))
        print  (utils.textcolor_display ('the experiment has done!', type_mes ='inf')      )     
        return time_text
    
def deepmg_visual(options,args):
    '''
    visualize data only
    '''
    #declare varibables for checking points
    time_text = str(strftime("%Y%m%d_%H%M%S", gmtime())) 
    start_time = time.time() # check point the beginning
    mid_time = [] # check point middle time to measure distance between runs

    #input shape
    input_shape = (options.dim_img, options.dim_img, options.channel)

      
    # ===================== STEP 1: NAMING folders and file pattern for outputs ===================
    # =============================================================================================
       
    path_write_parentfolder_img, res_dir , pre_name = naming_folder(options)  
      
    # !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=
    # !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!= END : NAMING folders and file pattern for outputs >!=!=!=!=!=!=


    # ===================== STEP 2: read orginal data from excel ===================
    # ==============================================================================
     
    data, labels, name_features = read_data(options,get_names_of_features = 'y')      
    if options.test_ext == 'y': #if use external validation set
        v_data_ori, v_labels = read_data(options,suffix='z',get_names_of_features = 'n' , mode = 'external')
    
    if options.del0 == 'y': # Delete column if it is All zeros (0)
        #print  (data.shape
        #print  ('delete columns which have all = 0'
        #data[:, (data != 0).any(axis=0)]
        data = data[:, (data != 0).any(axis=0)]
        #print  (data.shape   
        if options.test_ext == 'y':
            v_data_ori = v_data_ori[:,(v_data_ori != 0).any(axis=0)]
    
    if options.test_ext == 'y': #if use external validation set, check #features of internal and external       
        if v_data_ori.shape[1] != data.shape[1]:
            print  ('external and internal sets do not have the same of features, ('+ utils.textcolor_display(str(v_data_ori.shape[1]) + ' != '+str(data.shape[1]))+")")
            print  ( utils.textcolor_display ('Please check option '+utils.textcolor_display('--del0')+ 'or your data'))
            exit()
            
    # !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=
    # !=!=!=!=!=!=!=!=!=!=!=!=!=!= end of read orginal data from excel !=!=!=!=!=!=!=
   
    #building cordinates for fill-up 
    # (a square of len_square*len_square containing all values of data)        
    

    if options.fig_size == 0: #if fig_size <=0, use the smallest size which fit the data, generating images of (len_square x len_square)
        fig_v_size =  int(math.ceil(math.sqrt(data.shape[1])))  
    else:
        fig_v_size =  options.fig_size

    #create full set of images for fill-up using predefined-bins
    if options.type_emb in [ 'fill','fills','zfill','zfills'] and options.algo_redu in  ["",'none']: 
        #if use dimension reduction (algo_redu!=""), only applying after reducing
        
        if options.type_emb in ['zfill','zfills']:
            cordi_x, cordi_y = vis_data.coordinates_z_fillup(data.shape[1])    
        else:
            cordi_x, cordi_y = vis_data.coordinates_fillup(data.shape[1])   
        #create full set of images for fill-up using predefined-bins
        if options.type_emb in ['fill','zfill']: 
                
            data = create_fillup_folder (options,path_write_parentfolder_img,cordi_x,cordi_y,
                    data,labels, prefix_img = 'fill', only_visual = 'y')   
            if options.test_ext == 'y': #if use external validation set           
                v_data_ori = create_fillup_folder (options,path_write_parentfolder_img,cordi_x,cordi_y,
                    v_data_ori,v_labels, prefix_img = 'testval', only_visual = 'y')   

        #print  (utils.textcolor_display ('images were created successfully at ' + path_write_parentfolder_img,type_mes ='inf')
        exit()         

     

    if options.save_para == 'y' or options.type_run in ['config']: #IF save para to config file to rerun     
        configfile_w = ''     
        if options. path_config_w == "":
            #configfile_w = res_dir + "/" + pre_name + "_" + name2_para + ".cfg"
            configfile_w = res_dir + "/" + prefix + ".cfg"
        else: #if user want to specify the path
            #configfile_w = options. path_config_w + "/" + pre_name + "_" + name2_para + ".cfg"
            configfile_w = options.path_config_w  + "/" + prefix + ".cfg"

        #if os.path.isfile(configfile_w):
        #    #print  ('the config file exists!!! Please remove it, and try again!'
        #    exit()
        ##print  (options
        #else:
        write_para(options,configfile_w)  
        #print  ('Config file was saved to ' + utils.textcolor_display(configfile_w , type_mes = 'inf')  
        if options.type_run in ['config']:
            exit()

      


    # ===================== BEGIN: run each run ================================================
    # ==========================================================================================
    
    for seed_value in range(options.seed_value_begin,options.seed_value_begin+options.run_time) :
        #print  ("We're on seed %d" % (seed_value)    
        np.random.seed(seed_value) # for reproducibility    
        
        #begin each fold
        skf=StratifiedKFold(n_splits=options.n_folds, random_state=seed_value, shuffle= True)
        for index, (train_indices,val_indices) in enumerate(skf.split(data, labels)):
            if options.type_run in ['vis','visual']:
                print  ("Creating images on  " + str(seed_value) + ' fold' + str(index+1) + "/"+str(options.n_folds)+"..."  )     
            if options.debug == 'y':
                print(train_indices)
                print(val_indices)         

            #transfer data to train/val sets
            train_x = []
            train_y = []
            val_x = []
            val_y = []   
            train_x, val_x = data[train_indices], data[val_indices]
            train_y, val_y = labels[train_indices], labels[val_indices]
            if options.test_ext == 'y':  
                v_data =  v_data_ori
            #print  (train_x.shape
            ##### reduce dimension 
            if not (options.algo_redu in  ['','none']): #if use reducing dimension
                train_x, val_x, v_data = reduce_dim (options,prefix_file_log, seed_value, index, train_x, val_x, v_data)
                #apply cordi to new size
                #when appyling to reduce dimension, no more data which still = 0, so should not use type_dat='pr'
                if options.type_emb in [ 'fills','zfills']:
                    if options.type_emb in [ 'zfills']:
                        cordi_x, cordi_y = vis_data.coordinates_z_fillup(options.new_dim)   
                    else:
                        cordi_x, cordi_y = vis_data.coordinates_fillup(options.new_dim)   
           
            #print(train_x.shape)
            #print(val_x.shape)

            if options.emb_data != '': #if use original for embedding
                train_x_original = train_x      

           ##### scale mode, use for creating images/bins
            if options.scale_mode in ['none','']:
                print  ('do not use scaler')
            elif (options.scale_mode in ['mms','none','','qtf','qtnf','ln','ptf']) or options.scale_mode[0:3] == 'log' :
                train_x,val_x,v_data = scaler_section (options,prefix_file_log,seed_value,index,train_x,val_x,v_data)              
            else:
                print  ( utils.textcolor_display('this scaler (' + str(options.scale_mode) + ') is not supported now! Please try again!!'))
                exit()

            #select type_emb in ['raw','bin','fills','tsne',...]
            if options.type_emb == "raw" or options.type_emb == "bin": #or options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']: #reshape raw data into a sequence for ltsm model. 
                
                if options.type_emb == "bin" : 
                    if options.auto_v  == 'y': #if auto, adjust automatically min_v and max_v for binning
                        #options.min_v = np.min(train_x)
                        #options.max_v = np.max(train_x)
                        min_v_temp = np.min(train_x)
                        max_v_temp = np.max(train_x)
                    else:
                        min_v_temp = options.min_v
                        max_v_temp = options.max_v
                    
                    temp_train_x=[]
                    temp_val_x=[]
                    temp_v_data=[]


                    #build bins and binning
                    bins_breaks = vis_data.build_breaks (data=train_x, type_bin=options.type_bin, num_bin=options.num_bin, max_v= max_v_temp, min_v = min_v_temp)

                    for i in range(0, len(train_x)):   
                        temp_train_x.append( [vis_data.convert_bin(value=y, min_v = min_v_temp, num_bin = options.num_bin, bins_break = bins_breaks) for y in train_x[i] ])
                    train_x = np.stack(temp_train_x)
                    for i in range(0, len(val_x)):   
                        temp_val_x.append( [vis_data.convert_bin(value=y, min_v = min_v_temp, num_bin = options.num_bin, bins_break = bins_breaks) for y in val_x[i] ])
                    val_x = np.stack(temp_val_x)

                    if options.test_ext == 'y':     
                        for i in range(0, len(v_data)):   
                            temp_v_data.append( [vis_data.convert_bin(value=y,  min_v = min_v_temp,  num_bin = options.num_bin, bins_break = bins_breaks) for y in v_data[i] ])
                        v_data = np.stack(temp_v_data)
                                      

                #print  ('use data...' + str(options.type_emb )
                if options.model in ['model_con1d_baseline' , 'model_cnn1d']:
                    input_shape=((train_x.shape[1]),1)
                    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1],1))
                    val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1],1))
                    if options.test_ext == 'y':     
                        v_data = np.reshape(v_data, (v_data.shape[0], v_data.shape[1],1))

                elif options.model in [ 'svm_model','rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                    input_shape=((train_x.shape[1]))
                    print('shape==')
                    print((train_x.shape[0], train_x.shape[1]))
                    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))
                    val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1]))
                    if options.test_ext == 'y':     
                        v_data = np.reshape(v_data, (v_data.shape[0], v_data.shape[1]))
                else:
                    input_shape=(1,(train_x.shape[1]))
                    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
                    val_x = np.reshape(val_x, (val_x.shape[0], 1, val_x.shape[1]))      
                    if options.test_ext == 'y':     
                        v_data = np.reshape(v_data, (v_data.shape[0], 1, v_data.shape[1])) 
               
                #print('input_shape='+str(input_shape))
                ##print  ('train_x:=' + str(train_x.shape)
                
               
                    
            elif not (options.type_emb in ["fill",'zfill']): #embedding type, new images after each k-fold

                #save information on bins
               
                if options.auto_v  == 'y': #if auto, adjust automatically min_v and max_v for binning
                    #options.min_v = np.min(train_x)
                    #options.max_v = np.max(train_x)
                    min_v_temp = np.min(train_x)
                    max_v_temp = np.max(train_x)
                else:
                    min_v_temp = options.min_v
                    max_v_temp = options.max_v
                
                title_cols = np.array([["bins for classification"]])           
                np.savetxt(f,title_cols, fmt="%s",delimiter="/")               
                #save information on histogram of bins
                w_interval = float(max_v_temp - min_v_temp) / options.num_bin
                binv0 = []
                for ik in range(0,options.num_bin+1):
                    binv0.append(min_v_temp + w_interval*ik )


               
                # ++++++++++++ start embedding ++++++++++++

                #train_x, val_x, v_data = visual_section (options, path_write_parentfolder_img, seed_value, 
                #    train_x,val_x, train_y, val_y, v_data, v_labels,train_x_original,cordi_x, cordi_y , prefix_file_log)
                train_x, val_x, v_data = visual_section (options = options,  train_x=train_x,val_x=val_x, train_y=train_y, val_y=val_y, v_data=v_data, 
                    v_labels=v_labels,train_x_original=train_x_original,
                    prefix_file_log=prefix_file_log, path_write_parentfolder_img=path_write_parentfolder_img, 
                    cordi_x=cordi_x,cordi_y=cordi_y,train_test_whole = 'n',seed_value=seed_value, index=index)
                
                #!=!=!=!=!=!= END of read images for train/val sets !=!=!=!=!=!=
                # !=!=!=!=!=!= END of embedding !=!=!=!=!=!=
               
            if options.debug == 'y':
                print(train_x.shape)
                print(val_x.shape)
                print(train_y)
                print(val_y)  
                print("size of data: whole data")
                print(data.shape)
                print("size of data: train_x")
                print(train_x.shape)
                print("size of data: val_x")
                print(val_x.shape)
                print("size of label: train_y")
                print(train_y.shape)
                print("size of label: val_y")
                print(val_y.shape)    

           
                            
            if options.dim_img==-1 and options.type_emb != 'raw' and options.type_emb != 'bin': #if use real size of images
                input_shape = (train_x.shape[1],train_x.shape[2], options.channel)
            
            np.random.seed(seed_value) # for reproducibility   

            if options.type_run  in ['vis','visual']:  
                print  ('mode: VISUALIZATIONS, CREATING IMAGES, NOT LEARNING')
            else:
                print  ('mode: LEARNING')
                #tf.set_random_seed(seed_value)     
                try: 
                    tf.set_random_seed(seed_value)
                except Exception as exception:
                    tf.random.set_seed(seed_value)
                #rn.seed(seed_value) 
                      
                if options.e_stop==-1: #=-1 : do not use earlystopping
                    options.e_stop = options.epoch

                if options.e_stop_consec=='consec': #use sefl-defined func
                    early_stopping = models_def.EarlyStopping_consecutively(monitor='val_loss', 
                        patience=options.e_stop, verbose=1, mode='auto')
                else: #use keras func
                    early_stopping = kr.callbacks.EarlyStopping(monitor='val_loss', 
                        patience=options.e_stop, verbose=1, mode='auto')

                if options.coeff == 0: #run tuning for coef
                    print  (utils.textcolor_display ('Error! options.coeff == 0 only use for tuning coef....'))
                    exit()                                         
                        
                #+++++++++++ transform or rescale before fetch data into learning  +++++++++++
                #if options.num_classes >= 2:
                #    train_y=kr.utils.np_utils.to_categorical(train_y , num_classes = num_classes )
                #    val_y = kr.utils.np_utils.to_categorical(val_y , num_classes = num_classes )

                if options.debug == 'y':
                    print  ('max_before coeff=======')
                    print  (np.amax(train_x))
                train_x = train_x / float(options.coeff)
                val_x = val_x / float(options.coeff)   
                if options.test_ext == 'y':       
                    v_data = v_data/float(options.coeff)
                
                if options.num_classes >= 2:
                    train_y = kr.utils.np_utils.to_categorical(train_y,  num_classes = options.num_classes )
                    val_y = kr.utils.np_utils.to_categorical(val_y ,  num_classes = options.num_classes )
                    if options.test_ext == 'y':
                        v_labels = kr.utils.np_utils.to_categorical(v_labels ,  num_classes = options.num_classes )
                
                if options.debug == 'y':
                    print  ('max_after coeff======='    )             
                    print  ('train='+str(np.amax(train_x)))
                    if options.test_ext == 'y':       
                        print  ('val_test='+str(np.amax(v_data)))
                #!=!=!=!=!=!= end of transform or rescale before fetch data into learning  !=!=!=!=!=!=


                #++++++++++++   selecting MODELS AND TRAINING, TESTING ++++++++++++
               
                
                if options.model in  ['resnet50', 'vgg16']:
                    input_shape = Input(shape=(train_x.shape[1],train_x.shape[2], options.channel)) 
                
                #if options.seed_ml == 'y':
                #    seed_algo = seed_value
                #else:
                #    seed_algo = None

                model = models_def.call_model (type_model=options.model, 
                        m_input_shape= input_shape, m_num_classes= options.num_classes, m_optimizer = options.optimizer,
                        m_learning_rate=options.learning_rate, m_learning_decay=options.learning_rate_decay, m_loss_func=options.loss_func,
                        ml_number_filters=options.numfilters,ml_numlayercnn_per_maxpool=options.numlayercnn_per_maxpool, ml_dropout_rate_fc=options.dropout_fc, 
                        mc_nummaxpool=options.nummaxpool, mc_poolsize= options.poolsize, mc_dropout_rate_cnn=options.dropout_cnn, 
                        mc_filtersize=options.filtersize, mc_padding = options.padding,
                        svm_c=options.svm_c, svm_kernel=options.svm_kernel,rf_n_estimators=options.rf_n_estimators,
                        rf_max_depth = options.rf_max_depth,
                        knn_n_neighbors=options.knn_n_neighbors,
                        rf_max_features= options.rf_max_features
                        #, rf_random_state = seed_algo
                        )
                        
                if options.visualize_model == 'y':  #visualize architecture of model  
                    from keras_sequential_ascii import sequential_model_to_ascii_#printout
                    sequential_model_to_ascii_#printout(model)                      

                np.random.seed(seed_value) # for reproducibility     
                #tf.set_random_seed(seed_value)  
                try: 
                    tf.set_random_seed(seed_value)
                except Exception as exception:
                    tf.random.set_seed(seed_value) 
                #rn.seed(seed_value)                  
                        
                if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                    #print  (train_x.shape           
                    #print  ('run classic learning algorithms'  
                    model.fit(train_x, train_y)

                    if options.model in ['rf_model','dtc_model','gbc_model','svm_model']: #save important and scores of features in Random Forests                    
                        if  options.save_rf == 'y':
                            importances = model.feature_importances_
                            ##print  (importances
                            #indices_imp = np.argsort(importances)
                            #indices_imp = np.argsort(importances)[::-1] #reserved order
                            f=open(prefix_details+"s"+str(seed_value)+"k"+str(index+1)+"_importance_fea.csv",'a')
                            np.savetxt(f,np.c_[(name_features,importances)], fmt="%s",delimiter=",")
                            f.close()
                else:
                    #print  ('run deep learning algorithms'
                    history_callback=model.fit(train_x, train_y, 
                        epochs = options.epoch, 
                        batch_size=options.batch_size, verbose=1,
                        validation_data=(val_x, val_y), callbacks=[early_stopping],
                        shuffle=False)        # if shuffle=False could be reproducibility
                    #print  (history_callback
           

            if options.type_run in ['vis','visual']: #only visual, not learning
                print  ('mode: VISUALIZATIONS, CREATING IMAGES, NOT LEARNING, skip writing logs.... each fold')

            
        
            # !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!= finish one fold  !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=
            
        
        if options.type_run in ['vis','visual']: #only visual, not learning
            print  ('mode: VISUALIZATIONS, CREATING IMAGES, NOT LEARNING, skip writing logs.... each run time')

            
    
    if options.type_run in ['vis','visual']: #only visual, not learning
        print  ('mode: VISUALIZATIONS, CREATING IMAGES, NOT LEARNING, skip writing logs.... skip collecting results')
        print  ('images were created successfully at ' + path_write)

def run_holdout_deepmg(options,args, special_usecase = '', txt_time_pre =''):
    '''
    run holdout validation
    options, args (array): options provided from cmd or config file
    special_usecase (string): if do not use holdout (if test_size = 1 or 0), 
                        then can use either:
                             'external_validation' (activated from another run) or 
                             'predict' just for predict
                             'train_test_whole' train and test whole dataset, to get weights on whole dataset
    txt_time_pre (string): string of time to name log file (only used for external validation)
    '''
    if options.type_run not in ['vis','visual']:  
        import tensorflow as tf
        #import models_def #models definitions
        from deepmg import models_def
        import keras as kr
        from keras.applications.resnet50 import ResNet50
        from keras.applications.vgg16 import VGG16
        from keras.models import Model
        from keras.applications.resnet50 import preprocess_input
        from keras.utils import np_utils
        from keras import backend as optimizers
        from keras.models import Sequential
        from keras.layers import Activation, Dropout, Flatten, Dense, InputLayer, Conv2D, MaxPooling2D,Input
    
    if special_usecase in ['external_validation', 'predict','train_test_whole']: #if external_validation, predict or save_entire_w=y
        #options.run_time = 1
        if special_usecase in ['external_validation','train_test_whole']:
            options.test_size = 0
        else:
            options.test_size = 1


    #declare varibables for checking points
    time_text = str(strftime("%Y%m%d_%H%M%S", gmtime())) 
    start_time = time.time() # check point the beginning
    mid_time = [] # check point middle time to measure distance between runs

    #input shape
    input_shape = (options.dim_img, options.dim_img, options.channel)
      
    # ===================== STEP 1: NAMING folders and file pattern for outputs ===================
    # =============================================================================================
    
    
    
    #get the pattern name for files log1,2,3
    #if special_usecase in ['external_validation']:  
    #    prefix = name_log_final(options, n_time_text= str(txt_time_pre)+ '_extval')
    #    path_write_parentfolder_img, res_dir , pre_name = naming_folder(options)
    if special_usecase in ['train_test_whole']:
        prefix = name_log_final(options, n_time_text= str(txt_time_pre)+ '_whole')
        path_write_parentfolder_img, res_dir , pre_name = naming_folder(options, search_res = 'n')
    else:
        prefix = name_log_final(options, n_time_text= str(time_text))
        path_write_parentfolder_img, res_dir , pre_name = naming_folder(options)

    name2_para = name_log_final(options, n_time_text= '')
    ##print  (prefix    
    if options.debug == 'y':
        print  (prefix) 
    #file1: contained general results, each fold; file2: accuracy,loss each run (finish a 10-cv), file3: Auc
    #folder details: acc,loss of each epoch, folder models: contain file of model (.json)
    
   
    prefix_file_log = res_dir + "/" + prefix + "file"
    prefix_details = res_dir + "/details/" + prefix+"acc_loss_"
    prefix_models = res_dir + "/models/" + prefix+"model_"

    # !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=
    # !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!= END : NAMING folders and file pattern for outputs >!=!=!=!=!=!=


    # ===================== STEP 2: read orginal data from excel ===================
    # ==============================================================================         

    if special_usecase in ['predict'] and not os.path.isfile(os.path.join(options.original_data_folder , options.data_name + '_y.csv')):
        data, labels, name_features = read_data(options,included_labels='n',get_names_of_features = 'y')      
    else:           
        data, labels, name_features = read_data(options,get_names_of_features = 'y')      
    
    v_data_ori = []
    v_labels = []
    if options.test_ext == 'y': #if use external validation set
        v_data_ori, v_labels = read_data(options,suffix='z',get_names_of_features = 'n' , mode = 'external')

    if options.del0 == 'y':
        #print  (data.shape
        #print  ('delete columns which have all = 0'
        #data[:, (data != 0).any(axis=0)]
        data = data[:, (data != 0).any(axis=0)]
        #print  (data.shape       
        if options.test_ext == 'y':
            v_data_ori = v_data_ori[:,(v_data_ori != 0).any(axis=0)]
    
    if options.test_ext == 'y': #if use external validation set, check #features of internal and external       
        if v_data_ori.shape[1] != data.shape[1]:
            print  ('external and internal sets do not have the same of features, ('+ utils.textcolor_display(str(v_data_ori.shape[1]) + ' != '+str(data.shape[1]))+")")
            print  ( utils.textcolor_display ('Please check option '+utils.textcolor_display('--del0')+ 'or your data'))
            exit()
    # !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=
    # !=!=!=!=!=!=!=!=!=!=!=!=!=!= end of read orginal data from excel !=!=!=!=!=!=!=
   
    #building cordinates for fill-up 
    # (a square of len_square*len_square containing all values of data)
     #create full set of images for fill-up using predefined-bins
    cordi_x=[]
    cordi_y=[]

    if options.fig_size == 0: #if fig_size <=0, use the smallest size which fit the data, generating images of (len_square x len_square)
        fig_v_size =  int(math.ceil(math.sqrt(data.shape[1])))  
    else:
        fig_v_size =  options.fig_size

    if options.type_emb in [ 'fill','fills','zfill','zfills'] and options.algo_redu in  ["",'none']: 
        #if use dimension reduction (algo_redu!=""), only applying after reducing
        
        if options.type_emb in ['zfill','zfills']:
            cordi_x, cordi_y = vis_data.coordinates_z_fillup(data.shape[1])    
        else:
            cordi_x, cordi_y = vis_data.coordinates_fillup(data.shape[1])   
        #create full set of images for fill-up using predefined-bins
        if options.type_emb in ['fill','zfill']: 
             
            data = create_fillup_folder (options,path_write_parentfolder_img,cordi_x,cordi_y,
                    data,labels, prefix_img = 'fill')
            if options.test_ext == 'y': #if use external validation set           
                v_data_ori = create_fillup_folder (options,path_write_parentfolder_img,cordi_x,cordi_y,
                    v_data_ori,v_labels, prefix_img = 'testval')        
   
    #if use external validations, CREATING LOG FILE 4
    if options.test_ext == 'y':        
        
        #title_cols = np.array([["seed","fold","val_acc",'val_auc','val_mcc',"external_acc","external_auc","external_mcc"]])         
        #title_cols = np.array([["train_acc","train_auc","train_loss","train_mcc","val_acc","val_auc","val_loss","val_mcc","tn", "fp", "fn", "tp",
        #    "index_fold","seed"]])  #folds
        #title_cols = np.array([["val_acc","val_auc","val_loss","val_mcc","tn", "fp", "fn", "tp",
        #    "index_fold","seed"]])    
        title_cols = np.array([["train_acc","val_acc","train_auc","val_auc","train_loss","val_loss","val_mcc","tn", "fp", "fn", "tp",
            "time","ep_stopped", "index_fold","seed"]])
        #in external evaluation:
        # # train ---> val (in internal set)
        # # val --> test (in external set)
        f=open(prefix_file_log+"_ext.txt",'a')
        np.savetxt(f,title_cols, fmt="%s",delimiter="\t")
        f.close()

    #save arg/options/configurations used in the experiment to LOG file1
    f = open(prefix_file_log+"_sum.txt",'a')
    np.savetxt(f,(options, args), fmt="%s", delimiter="\t")
    f.close()

    if options.save_para == 'y' or options.type_run in ['config']: #IF save para to config file to rerun   
        configfile_w = ''       
        if options. path_config_w == "":
            configfile_w = res_dir + "/" + prefix + ".cfg"
            #configfile_w = res_dir + "/" + pre_name + "_" + name2_para + ".cfg"
        else: #if user want to specify the path
            #configfile_w = options. path_config_w + "/" + pre_name + "_" + name2_para + ".cfg"
            configfile_w = options.path_config_w + "/" + prefix + ".cfg"

        #if os.path.isfile(configfile_w):
        #    #print  ('the config file exists!!! Please remove it, and try again!'
        #    exit()
        ##print  (options
        #else:
        write_para(options,configfile_w)  
        #print  ('Config file was saved to ' + utils.textcolor_display(configfile_w , type_mes = 'inf')   
        if options.type_run in ['config']:
            exit()
      

    if options.save_avg_run in ['y']:
        #save title (column name) for log2: acc,loss each run with mean(k-fold-cv)
        title_cols = np.array([["seed","ep","train_acc_start","train_acc_end","val_acc_begin","val_acc_end","train_loss_start","train_loss_end","val_loss_start","val_loss_end""total_time","time_distant"]])  
    
        #if options.save_l in ['y']:
        #    title_cols = np.array([["seed","ep","tr_start","tr_fin","te_begin","te_fin","lo_tr_be","lo_tr_fi","lo_te_be","lo_te_fi""total_t","t_distant"]])  
        #else:
        #    title_cols = np.array([["seed","ep","train_start","train_fin","test_start","test_fin","total_t","t_distant"]])  
        
        f=open(prefix_file_log+"_mean_acc.txt",'a')
        np.savetxt(f,title_cols, fmt="%s",delimiter="\t")
        f.close()
        
        #save title (column name) for log3: acc,auc, confusion matrix each run with mean(k-fold-cv)
        title_cols = np.array([["se","ep","tr_tn","tr_fn","tr_fp","tr_tp","va_tn","va_fn","va_fp","va_tp","tr_acc",
            "va_acc","tr_auc","va_auc"]])  
        f=open(prefix_file_log+"_mean_auc.txt",'a')
        np.savetxt(f,title_cols, fmt="%s",delimiter="\t")
        f.close()

    ## initialize variables for global acc, auc
    temp_all_acc_loss=[]
    temp_all_auc_confusion_matrix=[]    

    train_acc_all =[] #stores the train accuracies of all folds, all runs
    train_auc_all =[] #stores the train auc of all folds, all runs
    #train_mmc_all =[] #stores the train auc of all folds, all runs
    val_mmc_all =[] #stores the validation accuracies of all folds, all runs
    val_acc_all =[] #stores the validation accuracies of all folds, all runs
    val_auc_all =[] #stores the validation auc of all folds, all runs
    
    best_mmc_allfolds_run = -1
    best_mmc_allfolds_run_get = -1
    best_acc_allfolds_run = 0
    best_auc_allfolds_run = 0

    # ===================== BEGIN: run each run ================================================
    # ==========================================================================================
    for seed_value in range(options.seed_value_begin,options.seed_value_begin+options.whole_run_time) :
        #print  ("We're on seed %d" % (seed_value)    
        np.random.seed(seed_value) # for reproducibility    
        if options.type_run not in ['vis','visual']:   
            try: 
                tf.set_random_seed(seed_value)
            except Exception as exception:
                tf.random.set_seed(seed_value)
        #rn.seed(seed_value)       

        #ini variables for scores in a seed
        #TRAIN: acc (at beginning + finish), auc, loss, tn, tp, fp, fn
        train_acc_1se=[]
        train_auc_1se=[]
        train_loss_1se=[]
        train_acc_1se_begin=[]
        train_loss_1se_begin=[]

        train_tn_1se=[]
        train_tp_1se=[]
        train_fp_1se=[]
        train_fn_1se=[]

        early_stop_1se=[]
       
        #VALIDATION: acc (at beginning + finish), auc, loss, tn, tp, fp, fn
        val_acc_1se=[]
        val_auc_1se=[]
        val_loss_1se=[]
        val_acc_1se_begin=[]
        val_loss_1se_begin=[]
        val_pre_1se=[]
        val_recall_1se=[]
        val_f1score_1se=[]
        val_mmc_1se=[]

        val_tn_1se=[]
        val_tp_1se=[]
        val_fp_1se=[]
        val_fn_1se=[]       

        #set the time distance between 2 runs, the time since the beginning
        distant_t=0
        total_t=0       

        train_x = []
        train_y = []
        val_x = []
        val_y = []  
        v_data = []

        
        if options.test_size == 0: #if evaluate on external validation, get whole data for training, get whole v_data_ori for testing
            #if special_usecase in ['train_test_whole']: #if train to get weight on whole dataset
            train_x = data
            train_y = labels
            val_x = data
            val_y = labels
            if options.test_ext == 'y':
                v_data = v_data_ori
            #else:
            #    train_x = data
            #    train_y = labels
            #    val_x = v_data
            #    val_y = v_labels

        elif options.test_size == 1: #if mode of "predict", get whole data for testing, in the case no learning, just using pretrained weights from file
            train_x = data
            train_y = labels
            val_x = data
            val_y = labels

        else:
            train_x, val_x, train_y, val_y = train_test_split(data, labels, test_size= options.test_size, random_state=seed_value)

        
        train_x_original = train_x
        v_val_auc_score = -1
        
        for index in range(0,1):
            
            if not (options.algo_redu in  ['','none']): #if use reducing dimension
                train_x, val_x, v_data = reduce_dim (options,prefix_file_log, seed_value, index, train_x, val_x, v_data)
                #apply cordi to new size
                #when appyling to reduce dimension, no more data which still = 0, so should not use type_dat='pr'
                if options.type_emb in [ 'fills']:                    
                    cordi_x, cordi_y = vis_data.coordinates_fillup(options.new_dim)         
                elif options.type_emb in [ 'zfills']: 
                    cordi_x, cordi_y = vis_data.coordinates_z_fillup(options.new_dim)         

            if options.emb_data != '': #if use original for embedding
                train_x_original = train_x 

            ##### scale mode, use for creating images/bins
            if options.scale_mode in ['none','']:
                print  ('do not use scaler')
            elif (options.scale_mode in ['mms','none','','qtf','qtnf','ln','ptf']) or options.scale_mode[0:3] == 'log' :
                train_x,val_x,v_data = scaler_section (options,prefix_file_log,seed_value,index,train_x,val_x,v_data)              
            else:
                print  ( utils.textcolor_display ('this scaler (' + str(options.scale_mode) + ') is not supported now! Please try again!!'))
                exit()

            #select type_emb in ['raw','bin','fills','tsne',...]
            if options.type_emb == "raw" or options.type_emb == "bin": # or options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']: #reshape raw data into a sequence for ltsm model. 
                
                if options.type_emb == "bin" : 
                    if options.auto_v  == 'y': #if auto, adjust automatically min_v and max_v for binning
                        #options.min_v = np.min(train_x)
                        #options.max_v = np.max(train_x)
                        min_v_temp = np.min(train_x)
                        max_v_temp = np.max(train_x)
                    else:
                        min_v_temp = options.min_v
                        max_v_temp = options.max_v
                    
                    temp_train_x=[]
                    temp_val_x=[]
                    temp_v_data=[]
                    
                    #build bins and binning
                    bins_breaks = vis_data.build_breaks (data=train_x, type_bin=options.type_bin, num_bin=options.num_bin, max_v= max_v_temp, min_v = min_v_temp)

                    for i in range(0, len(train_x)):   
                        temp_train_x.append( [vis_data.convert_bin(value=y, min_v = min_v_temp, num_bin = options.num_bin, bins_break = bins_breaks) for y in train_x[i] ])
                    train_x = np.stack(temp_train_x)
                    for i in range(0, len(val_x)):   
                        temp_val_x.append( [vis_data.convert_bin(value=y, min_v = min_v_temp, num_bin = options.num_bin, bins_break = bins_breaks) for y in val_x[i] ])
                    val_x = np.stack(temp_val_x)

                    if options.test_ext == 'y':     
                        for i in range(0, len(v_data)):   
                            temp_v_data.append( [vis_data.convert_bin(value=y,  min_v = min_v_temp,  num_bin = options.num_bin, bins_break = bins_breaks) for y in v_data[i] ])
                        v_data = np.stack(temp_v_data)

                #print  ('use data...' + str(options.type_emb )
                if options.model in ['model_con1d_baseline' , 'model_cnn1d']:
                    input_shape=((train_x.shape[1]),1)
                    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1],1))
                    val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1],1))                    

                    if options.test_ext == 'y':     
                        v_data = np.reshape(v_data, (v_data.shape[0], v_data.shape[1],1))

                elif options.model in [ 'svm_model','rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                    input_shape=((train_x.shape[1]))
                    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1]))
                    val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1]))

                    if options.test_ext == 'y':     
                        v_data = np.reshape(v_data, (v_data.shape[0], v_data.shape[1]))
                    
                else:
                    input_shape=(1,(train_x.shape[1]))
                    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
                    val_x = np.reshape(val_x, (val_x.shape[0], 1, val_x.shape[1]))     

                    if options.test_ext == 'y':     
                        v_data = np.reshape(v_data, (v_data.shape[0], 1, v_data.shape[1]))
               
                #print('input_shape='+str(input_shape))
                ##print  ('train_x:=' + str(train_x.shape)
                
               
                    
            elif not (options.type_emb in ["fill",'zfill']): #embedding type, new images after each k-fold

                #save information on bins
                f=open(prefix_file_log+"_sum.txt",'a')          
                if options.auto_v  == 'y': #if auto, adjust automatically min_v and max_v for binning
                    #options.min_v = np.min(train_x)
                    #options.max_v = np.max(train_x)
                    min_v_temp = np.min(train_x)
                    max_v_temp = np.max(train_x)
                else:
                    min_v_temp = options.min_v
                    max_v_temp = options.max_v
                
                title_cols = np.array([["bins for classification"]])           
                np.savetxt(f,title_cols, fmt="%s",delimiter="/")               
                #save information on histogram of bins
                w_interval = float(max_v_temp - min_v_temp) / options.num_bin
                binv0 = []
                for ik in range(0,options.num_bin+1):
                    binv0.append(min_v_temp + w_interval*ik )

                np.savetxt(f,(np.histogram(val_x, bins = binv0), np.histogram(train_x, bins = binv0)), fmt="%s",delimiter="\n")
                f.close()

             
                # ++++++++++++ start embedding ++++++++++++

                #train_x, val_x, v_data = visual_section (options, path_write_parentfolder_img, seed_value, index,
                #    train_x,val_x, train_y, val_y, v_data, v_labels,train_x_original,cordi_x, cordi_y , prefix_file_log, train_test_whole = 'y')
                train_x, val_x, v_data = visual_section (options = options,  train_x=train_x,val_x=val_x, train_y=train_y, 
                    val_y=val_y, v_data=v_data, 
                    v_labels=v_labels,train_x_original=train_x_original,
                    prefix_file_log=prefix_file_log, path_write_parentfolder_img=path_write_parentfolder_img, 
                    cordi_x=cordi_x,cordi_y=cordi_y,train_test_whole = 'y',seed_value=seed_value, index=index)
                
                #!=!=!=!=!=!= END of read images for train/val sets !=!=!=!=!=!=
                # !=!=!=!=!=!= END of embedding !=!=!=!=!=!=
               
            if options.debug == 'y':
                print(train_x.shape)
                #print(val_x.shape)
                #print(train_y)
                #print(val_y)  
                #print("size of data: whole data")
                #print(data.shape)
                #print("size of data: train_x")
                #print(train_x.shape)
                #print("size of data: val_x")
                #print(val_x.shape)
                #print("size of label: train_y")
                #print(train_y.shape)
                #print("size of label: val_y")
                #print(val_y.shape)    

            f=open(prefix_file_log+"_sum.txt",'a')
            title_cols = np.array([["seed",seed_value," fold",index+1,"/",options.n_folds,"###############","seed",seed_value," fold",index+1,"/",options.n_folds,"###############","seed",seed_value," fold",index+1,"/",options.n_folds]])
            np.savetxt(f,title_cols, fmt="%s",delimiter="")
            
            if options.save_labels in ['y']:
                #save lables of train/test set to log
                

                title_cols = np.array([["train_set"]])
                np.savetxt(f,title_cols, fmt="%s",delimiter="")        
                np.savetxt(f,[(train_y)], fmt="%s",delimiter=" ")

                title_cols = np.array([["validation_set"]])
                np.savetxt(f,title_cols, fmt="%s",delimiter="")    
                np.savetxt(f,[(val_y)], fmt="%s",delimiter=" ")

                if options.test_ext == 'y':
                    title_cols = np.array([["external_set"]])
                    np.savetxt(f,title_cols, fmt="%s",delimiter="")    
                    np.savetxt(f,[(v_labels)], fmt="%s",delimiter=" ")
                f.close()   
                            
            if options.dim_img==-1 and options.type_emb != 'raw' and options.type_emb != 'bin': #if use real size of images
                input_shape = (train_x.shape[1],train_x.shape[2], options.channel)
            
            np.random.seed(seed_value) # for reproducibility   

            if options.type_run  in ['vis','visual']:  
                print  ('mode: VISUALIZATIONS, CREATING IMAGES, NOT LEARNING')
            else:
                print  ('mode: LEARNING')
                #tf.set_random_seed(seed_value)  
                try: 
                    tf.set_random_seed(seed_value)
                except Exception as exception:
                    tf.random.set_seed(seed_value)  

                #rn.seed(seed_value) 
                      
                if options.e_stop==-1: #=-1 : do not use earlystopping
                    options.e_stop = options.epoch

                if options.e_stop_consec=='consec': #use sefl-defined func
                    early_stopping = models_def.EarlyStopping_consecutively(monitor='val_loss', 
                        patience=options.e_stop, verbose=1, mode='auto')
                else: #use keras func
                    early_stopping = kr.callbacks.EarlyStopping(monitor='val_loss', 
                        patience=options.e_stop, verbose=1, mode='auto')

                if options.coeff == 0: #run tuning for coef
                    #print  ('Error! options.coeff == 0 only use for tuning coef....'
                    exit()                                         
                        
                #+++++++++++ transform or rescale before fetch data into learning  +++++++++++


                if options.debug == 'y':
                    print  ('max_before coeff=======')
                    print  (np.amax(train_x))
                train_x = train_x / float(options.coeff)
                val_x = val_x / float(options.coeff)   
                if options.test_ext == 'y':
                    v_data = v_data / float(options.coeff)   

                if options.num_classes >= 2:
                    train_y=kr.utils.np_utils.to_categorical(train_y,  num_classes = options.num_classes )
                    val_y = kr.utils.np_utils.to_categorical(val_y ,  num_classes = options.num_classes )
                    if options.test_ext == 'y':
                        v_labels = kr.utils.np_utils.to_categorical(v_labels ,  num_classes = options.num_classes )

                               
                if options.debug == 'y':
                    print  ('max_after coeff======='  )               
                    print  ('train='+str(np.amax(train_x)))                  
                      
                #!=!=!=!=!=!= end of transform or rescale before fetch data into learning  !=!=!=!=!=!=
                #++++++++++++   selecting MODELS AND TRAINING, TESTING ++++++++++++             
                
                if options.model in  ['resnet50', 'vgg16']:
                    input_shape = Input(shape=(train_x.shape[1],train_x.shape[2], options.channel)) 
                
                #np.random.seed(seed_value)
                #if options.seed_ml == 'y':
                #    seed_algo = seed_value
                #else:
                #    seed_algo = None

                model = models_def.call_model (type_model=options.model, 
                        m_input_shape= input_shape, m_num_classes= options.num_classes, m_optimizer = options.optimizer,
                        m_learning_rate=options.learning_rate, m_learning_decay=options.learning_rate_decay, m_loss_func=options.loss_func,
                        ml_number_filters=options.numfilters,ml_numlayercnn_per_maxpool=options.numlayercnn_per_maxpool, ml_dropout_rate_fc=options.dropout_fc, 
                        mc_nummaxpool=options.nummaxpool, mc_poolsize= options.poolsize, mc_dropout_rate_cnn=options.dropout_cnn, 
                        mc_filtersize=options.filtersize, mc_padding = options.padding,
                        svm_c=options.svm_c, svm_kernel=options.svm_kernel,
                        rf_n_estimators=options.rf_n_estimators,
                        rf_max_depth = options.rf_max_depth,
                        m_pretrained_file = options.pretrained_w_path,knn_n_neighbors=options.knn_n_neighbors,
                        rf_max_features= options.rf_max_features
                        #,                        rf_random_state = seed_algo
                        )
                   
                if options.visualize_model == 'y':  #visualize architecture of model  
                    from keras_sequential_ascii import sequential_model_to_ascii_#printout
                    sequential_model_to_ascii_#printout(model)                      

                np.random.seed(seed_value) # for reproducibility     
                #tf.set_random_seed(seed_value)   
                #rn.seed(seed_value)     
                # 
                try: 
                    tf.set_random_seed(seed_value)
                except Exception as exception:
                    tf.random.set_seed(seed_value)             

                ################## after evaluating, quit the package ########################
                if special_usecase in ['predict']:   
                    
                    Y_pred_v = model.predict(val_x)
                    Y_pred_v_proba = model.predict_proba(val_x)
                    #print  (Y_pred_v
                    #Y_pred_v = model.predict(val_x)
                    if options.debug == 'y':
                        print  (Y_pred_v)                   

                    if len( labels) > 0 :#if existing the label
                        v_val_acc_score = accuracy_score(val_y, Y_pred_v.round())
                        if options.num_classes > 2:
                            v_val_auc_score = -1
                            #v_val_auc_score = multiclass_roc_auc_score(val_y, Y_pred_v)
                            v_val_mmc_score = -2
                        else:
                            v_val_auc_score = roc_auc_score(val_y, Y_pred_v)
                            v_val_mmc_score = matthews_corrcoef(val_y, Y_pred_v.round())
                        cm = confusion_matrix(val_y,Y_pred_v.round())
                        if options.num_classes > 2: 
                            tn = -1
                            fp = -1
                            fn = -1
                            tp = -1
                        else:
                            tn, fp, fn, tp = cm.ravel()

                        f=open(prefix_file_log+"_sum.txt",'a')  
                        title_cols = np.array([["acc","auc","mcc","tn","fp","fn","tp"]])
                        np.savetxt(f,title_cols, fmt="%s",delimiter="\t") 
                        np.savetxt(f,np.c_[(v_val_acc_score,v_val_auc_score,v_val_mmc_score,tn,fp,fn,tp
                            )], fmt="%s",delimiter="\t")
                        f.close() 
                        #print  ('acc='+ str(v_val_acc_score)+ ' auc=' + str(v_val_auc_score)+ ' mcc=' + str(v_val_mmc_score)

                        f=open(prefix_file_log+"_predicted.txt",'w')  #create new
                        title_cols = np.array([["y_predicted",'rounded','y_ground_truth']])
                        np.savetxt(f,title_cols, fmt="%s",delimiter="\t") 
                        np.savetxt(f,np.c_[Y_pred_v_proba[:,1],Y_pred_v_proba[:,1].round(),val_y], fmt="%s",delimiter="\t")
                        f.close() 
                    else:
                        f=open(prefix_file_log+"_predicted.txt",'w')  #create new
                        title_cols = np.array([["y_predicted",'rounded']])
                        np.savetxt(f,title_cols, fmt="%s",delimiter="\t") 
                        np.savetxt(f,np.c_[Y_pred_v_proba[:,1],Y_pred_v_proba[:,1].round()], fmt="%s",delimiter="\t")
                        f.close() 
                    
                    #os.remove(prefix_file_log+"_mean_auc.txt") #remove unused file
                    #mark the experiment finished
                    if options.cudaid > -3:  
                    #if options.cudaid not in  ["-1",'']:  
                        #os.rename(res_dir + "/" + prefix+"file_sum.txt", res_dir + "/" + prefix+"gpu"+ str(options.cudaid) + "file_sum_"+str(options.suff_fini)+".txt")  
                        os.rename(res_dir + "/" + prefix+"file_sum.txt", res_dir + "/" + prefix+"gpu_file_sum_"+str(options.suff_fini)+".txt")  
                    else:
                        os.rename(prefix_file_log+"_sum.txt", prefix_file_log+"_sum_"+str(options.suff_fini)+".txt")  

                    exit() ##!=!=!=!=!=!=!=!=!=!=!= after evaluating, quit the package !=!=!=!=!=!=!=!=!=!=!=##
               

                if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                    #print  (train_x.shape           
                    #print  ('run classic learning algorithms'  
                    
                    model.fit(train_x, train_y)
             

                    if options.model in ['rf_model','dtc_model','gbc_model','svm_model']: #save important and scores of features in Random Forests                    
                        if  options.save_rf == 'y':
                            importances = model.feature_importances_
                            ##print  (importances
                            #indices_imp = np.argsort(importances)
                            #indices_imp = np.argsort(importances)[::-1] #reserved order
                            f=open(prefix_details+"s"+str(seed_value)+"k"+str(index+1)+"_importance_fea.csv",'a')
                            np.savetxt(f,np.c_[(name_features,importances)], fmt="%s",delimiter=",")
                            f.close()
                else:
                    #print  ('run deep learning algorithms'
                    history_callback=model.fit(train_x, train_y, 
                        epochs = options.epoch, 
                        batch_size=options.batch_size, verbose=1,
                        validation_data=(val_x, val_y), callbacks=[early_stopping],
                        shuffle=False)        # if shuffle=False could be reproducibility
                    #print  (history_callback
            #!=!=!=!=!=!= end of selecting MODELS AND TRAINING, TESTING  !=!=!=!=!=!=

          
            if options.type_run in ['vis','visual']: #only visual, not learning
                print  ('mode: VISUALIZATIONS, CREATING IMAGES, NOT LEARNING, skip writing logs.... each fold')

            else: #learning
                train_acc =[]
                train_loss = []
                val_acc = []
                val_loss = []

                if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
            
                    #ep_arr=range(1, 2, 1) #epoch   
                    ep_arr = []  
                    ep_arr.append(0)
                    ep_arr.append(1)

                    train_acc.append ( -1)
                    train_loss.append ( -1)
                    val_acc .append (-1)
                    val_loss .append ( -1)
                    

                    Y_pred = model.predict_proba(train_x)#, verbose=2)
                    #print  (model  
                    ##print  (Y_pred
                    #Y_pred = model.predict(train_x) #will check auc later
                    if options.debug == 'y':
                        print  (Y_pred)
                        print  (Y_pred[:,1])
                    ##print  (Y_pred.loc[:,0].values
                    ##print  (y_pred
                    if options.num_classes > 2:
                        train_auc_score = -1
                        #train_auc_score = multiclass_roc_auc_score(train_y, Y_pred[:,1])
                    else:
                        train_auc_score = roc_auc_score(train_y, Y_pred[:,1])       

                    if options.num_classes > 2:
                        train_acc.append(accuracy_score(train_y, Y_pred[:,1]))
                        train_loss.append(log_loss(train_y, Y_pred[:,1]))
                    else:
                        train_acc.append(accuracy_score(train_y, Y_pred[:,1].round()))
                        train_loss.append(log_loss(train_y, Y_pred[:,1]))
                    Y_pred_val = model.predict_proba(val_x)
                    val_acc.append ( accuracy_score(val_y, Y_pred_val[:,1].round()))
                    val_loss.append ( log_loss(val_y, Y_pred_val[:,1]))

                else:
                    #ep_arr=range(1, options.epoch+1, 1) #epoch
                    try:
                        ep_arr=range(1, len(history_callback.history['acc']), 1) #epoch
                        train_acc = history_callback.history['acc']  #acc
                        train_loss = history_callback.history['loss'] #loss
                        val_acc = history_callback.history['val_acc']  #acc
                        val_loss = history_callback.history['val_loss'] #loss
                    except:
                                            
                        ep_arr=range(1, len(history_callback.history['accuracy'])+1, 1) #epoch
                        train_acc = history_callback.history['accuracy']  #acc
                        train_loss = history_callback.history['loss'] #loss
                        val_acc = history_callback.history['val_accuracy']  #acc
                        val_loss = history_callback.history['val_loss'] #loss

                #store for every epoch
                title_cols = np.array(["ep","train_loss","train_acc","val_loss","val_acc"])  
                res=(ep_arr,train_loss,train_acc,val_loss,val_acc)
                res=np.transpose(res)
                combined_res=np.array(np.vstack((title_cols,res)))
                #combined_res=np.array((title_cols,res))

                #save to file, a file contains acc,loss of all epoch of this fold
                #print  ('prefix_details===='            
                #print  (prefix_details   
                if options.save_d in ['y'] :           
                    np.savetxt(prefix_details+"s"+str(seed_value)+"k"+str(index+1)+".txt", 
                        combined_res, fmt="%s",delimiter="\t")
                
                #!=!=!=!=!=!= END of SAVE ACC, LOSS of each epoch !=!=!=!=!=!=

                #++++++++++++ CONFUSION MATRIX and other scores ++++++++++++#
                #train
                ep_stopped = len(train_acc)
                
                #print  ('epoch stopped= '+str(ep_stopped)
                early_stop_1se.append(ep_stopped)

                train_acc_1se.append (train_acc[ep_stopped-1])
                train_acc_1se_begin.append (train_acc[0] )
                train_loss_1se.append (train_loss[ep_stopped-1])
                train_loss_1se_begin.append (train_loss[0] )

                if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                    Y_pred = model.predict_proba(train_x)#, verbose=2)
                    #Y_pred = model.predict(train_x) #will check auc later
                    if options.debug == 'y':
                        print  (Y_pred)
                        print  (Y_pred[:,1])
                    ##print  (Y_pred.loc[:,0].values
                    ##print  (y_pred
                    if options.num_classes > 2:
                        train_auc_score = -1
                        #train_auc_score = multiclass_roc_auc_score(train_y, Y_pred[:,1])
                    else:
                        train_auc_score = roc_auc_score(train_y, Y_pred[:,1])
                else:
                    Y_pred = model.predict(train_x, verbose=2)
                    if options.num_classes > 2:
                        train_auc_score = -1
                        #train_auc_score = multiclass_roc_auc_score(train_y, Y_pred[:,1])
                    else:
                        train_auc_score = roc_auc_score(train_y, Y_pred)
                
                #store to var, global acc,auc  
                

                if options.num_classes>=2:   
                    y_pred = np.argmax(Y_pred, axis=1)       
                    cm = confusion_matrix(np.argmax(train_y,axis=1),y_pred)
                    if options.num_classes > 2: 
                        tn = -1
                        fp = -1
                        fn = -1
                        tp = -1
                    else:
                        tn, fp, fn, tp = cm.ravel()                
                    #print("confusion matrix train")
                    #print(cm.ravel())
                else:
                    if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                        if options.debug == 'y':
                            print(Y_pred[:,1])
                        cm = confusion_matrix(train_y,Y_pred[:,1].round())
                        if options.num_classes > 2: 
                            tn = -1
                            fp = -1
                            fn = -1
                            tp = -1
                        else:
                            tn, fp, fn, tp = cm.ravel()
                    
                        #print("confusion matrix train")
                        #print(cm.ravel())

                       
                        
                    else:
                        if options.debug == 'y':
                            print(Y_pred.round())
                        cm = confusion_matrix(train_y,Y_pred.round())
                        if options.num_classes > 2: 
                            tn = -1
                            fp = -1
                            fn = -1
                            tp = -1
                        else:
                            tn, fp, fn, tp = cm.ravel()
                    
                        #print("confusion matrix train")
                        #print(cm.ravel())
                
                train_tn_1se.append (tn)
                train_fp_1se.append (fp)
                train_fn_1se.append (fn) 
                train_tp_1se.append (tp)
                train_auc_1se.append (train_auc_score)

                #test
                

                if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                    #Y_pred = model.predict(val_x)
                    Y_pred = model.predict_proba(val_x)#, verbose=2)
                    if options.debug == 'y':
                        print  ('scores from svm/rf')
                        print  (Y_pred)
                    if options.num_classes > 2:
                        val_auc_score = -1
                        #val_auc_score = multiclass_roc_auc_score(val_y, Y_pred[:,1])
                    else:
                        val_auc_score=roc_auc_score(val_y, Y_pred[:,1])
                else:
                    Y_pred = model.predict(val_x, verbose=2)
                    if options.debug == 'y':
                        print  ('scores from nn')
                        print  (Y_pred)
                    if options.num_classes > 2:
                        val_auc_score = -1
                        #val_auc_score = multiclass_roc_auc_score(val_y, Y_pred)
                    else:
                        val_auc_score=roc_auc_score(val_y, Y_pred)
                ##print(Y_pred)
                #print("score auc" +str(val_auc_score))          
                
                #store to var, global acc,auc  
            

                if options.num_classes>=2:     
                    y_pred = np.argmax(Y_pred, axis=1)    
                    cm = confusion_matrix(np.argmax(val_y,axis=1),y_pred)
                    if options.num_classes > 2: 
                        tn = -1
                        fp = -1
                        fn = -1
                        tp = -1
                    else:
                        tn, fp, fn, tp = cm.ravel()
                    #print("confusion matrix val")
                    #print(cm.ravel())
                else:
                    if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                        if options.debug == 'y':
                            print(Y_pred[:,1])
                        cm = confusion_matrix(val_y,Y_pred[:,1].round())
                        if options.num_classes > 2: 
                            tn = -1
                            fp = -1
                            fn = -1
                            tp = -1
                        else:
                            tn, fp, fn, tp = cm.ravel()
                    
                        #print("confusion matrix val")
                        #print(cm.ravel())
                        if options.num_classes > 2:
                            val_auc_score = -1
                            #val_auc_score = multiclass_roc_auc_score(val_y, Y_pred[:,1])
                            val_mmc = -2
                        else:
                            val_auc_score = roc_auc_score(val_y, Y_pred[:,1])
                            val_mmc = matthews_corrcoef(val_y,Y_pred[:,1].round())
                        val_acc.append ( accuracy_score(val_y, Y_pred[:,1].round()))
                        val_loss.append ( log_loss(val_y, Y_pred[:,1]))
                        
                    else:
                        if options.debug == 'y':
                            print(Y_pred.round())
                        cm = confusion_matrix(val_y,Y_pred.round())
                        if options.num_classes > 2: 
                            tn = -1
                            fp = -1
                            fn = -1
                            tp = -1
                        else:
                            tn, fp, fn, tp = cm.ravel()
                        #print("confusion matrix val")
                        #print(cm.ravel())
                if options.num_classes > 2:
                    val_mmc = -2
                else:
                    if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                        val_mmc = matthews_corrcoef(val_y,Y_pred[:,1].round())
                    else:
                        val_mmc = matthews_corrcoef(val_y,Y_pred.round())
                        
                
                val_acc_1se.append (val_acc[ep_stopped-1])
                val_acc_1se_begin.append (val_acc[0] )
                val_loss_1se.append (val_loss[ep_stopped-1])
                val_loss_1se_begin.append (val_loss[0] )

                val_tn_1se.append (tn)
                val_fp_1se.append (fp)
                val_fn_1se.append (fn) 
                val_tp_1se.append (tp)
                val_auc_1se.append (val_auc_score)

                # Accuracy = TP+TN/TP+FP+FN+TN
                # Precision = TP/TP+FP
                # Recall = TP/TP+FN
                # F1 Score = 2 * (Recall * Precision) / (Recall + Precision)
                # TP*TN - FP*FN / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

                if tp+fp == 0:
                    val_pre = 0
                else:
                    val_pre = np.around( tp/float(tp+fp),decimals=3)

                if tp+fn == 0:
                    val_recall=0
                else:
                    val_recall = np.around(tp/float(tp+fn),decimals=3)
                
                if val_recall+val_pre == 0:
                    val_f1score = 0
                else:
                    val_f1score = np.around(2 * (val_recall*val_pre)/ float(val_recall+val_pre),decimals=3)

                #if math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) == 0:
                #    val_mmc = 0
                #else:
                #    val_mmc = np.around(float(tp*tn - fp*fn) / float(math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))),decimals=3)
                
                
                #!=!=!=!=!=!= END of CONFUSION MATRIX and other scores  !=!=!=!=!=!=

                val_pre_1se.append (val_pre)
                val_recall_1se.append (val_recall)
                val_f1score_1se.append (val_f1score)
                val_mmc_1se.append (val_mmc)

                f=open(prefix_file_log+"_sum.txt",'a')   
                t_checked=time.time()           
                # title_cols = np.array([["seed",seed_value," fold",index+1,"/",options.n_folds]])
                # np.savetxt(f,title_cols, fmt="%s",delimiter="")
                title_cols = np.array([["after training"]])           
                np.savetxt(f,title_cols, fmt="%s",delimiter="\t")
                title_cols = np.array([["t_acc","v_acc","t_auc","v_auc","t_los","v_los",'val_mmc',"tn", "fp", "fn", "tp","time","ep_stopped"]])           
                np.savetxt(f,title_cols, fmt="%s",delimiter="--")
                np.savetxt(f,np.around(np.c_[( 
                    train_acc[ep_stopped-1],val_acc[ep_stopped-1] , 
                    train_auc_score,val_auc_score,                    
                    train_loss[ep_stopped-1] , val_loss[ep_stopped-1],
                    val_mmc,tn, fp, fn, tp,(t_checked - start_time),ep_stopped)
                    ], decimals=3), fmt="%s",delimiter="++")
                f.close() 

                train_acc_all.append(train_acc[ep_stopped-1])
                train_auc_all.append(train_auc_score)
                val_acc_all.append(val_acc[ep_stopped-1])
                val_auc_all.append(val_auc_score)
                val_mmc_all.append(val_mmc)          

                if options.test_ext == 'y':                      
                    #auc score, acc, mmc
                    if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                        Y_pred_v = model.predict_proba(v_data)#, verbose=2)               
                        if options.num_classes > 2:
                            v_val_auc_score = -1
                            #v_val_auc_score = multiclass_roc_auc_score(v_labels, Y_pred_v[:,1])
                            v_val_mmc_score = -2
                        else:
                            v_val_auc_score = roc_auc_score(v_labels, Y_pred_v[:,1])
                            v_val_mmc_score = matthews_corrcoef(v_labels, Y_pred_v[:,1].round())                        
                        v_val_loss_score = log_loss (v_labels, Y_pred_v[:,1])
                        v_val_acc_score = accuracy_score(v_labels, Y_pred_v[:,1].round())                    
                        
                    else:                                              
                        Y_pred_v = model.predict(v_data)     
                        if options.num_classes > 2:
                            v_val_auc_score = -1
                            #v_val_auc_score = multiclass_roc_auc_score(v_labels, Y_pred_v[:,1])
                            v_val_mmc_score = -2
                        else:                   
                            v_val_auc_score = roc_auc_score(v_labels, Y_pred_v)
                            v_val_mmc_score = matthews_corrcoef(v_labels, Y_pred_v.round())           
                        v_val_loss_score = log_loss (v_labels, Y_pred_v)
                        v_val_acc_score = accuracy_score(v_labels, Y_pred_v.round())                    
                                 

                    if options.debug == 'y':
                        print  (Y_pred)
                        
                    #confusion matrix
                    if options.num_classes>=2:     
                        y_pred = np.argmax(Y_pred_v, axis=1)    
                        cm = confusion_matrix(np.argmax(v_labels,axis=1),y_pred)
                        if options.num_classes > 2: 
                            tn = -1
                            fp = -1
                            fn = -1
                            tp = -1
                        else:
                            tn, fp, fn, tp = cm.ravel()
                        #print("confusion matrix external")
                        #print(cm.ravel())
                    else:
                        if options.model in ['svm_model', 'rf_model','dtc_model','gbc_model','knn_model'] or (options.model in [ 'pretrained'] and options.pretrained_w_path[len(options.pretrained_w_path)-2]+options.pretrained_w_path[len(options.pretrained_w_path)-1] in ['ml']):
                            if options.debug == 'y':
                                print(Y_pred_v[:,1])
                            cm = confusion_matrix(v_labels,Y_pred_v[:,1].round())
                            if options.num_classes > 2: 
                                tn = -1
                                fp = -1
                                fn = -1
                                tp = -1
                            else:
                                    tn, fp, fn, tp = cm.ravel()
                        
                            #print("confusion matrix external")
                            #print(cm.ravel())
                        else:
                            if options.debug == 'y':
                                print(Y_pred_v.round())
                            cm = confusion_matrix(v_labels,Y_pred_v.round())
                            if options.num_classes > 2: 
                                tn = -1
                                fp = -1
                                fn = -1
                                tp = -1
                            else:
                                tn, fp, fn, tp = cm.ravel()
                            #print("confusion matrix external")
                            #print(cm.ravel())
                    #title_cols = np.array([["val_acc","val_auc","val_loss","val_mcc","tn", "fp", "fn", "tp",
                    #        "index_fold","seed"]])  
                    f=open(prefix_file_log+"_ext.txt",'a')   
                    #title_cols = np.array([["train_acc","val_acc","train_auc","val_auc","train_loss","val_loss","val_mcc","tn", "fp", "fn", "tp",
                    #    "time","ep_stopped", "index_fold","seed"]])
                    #in external evaluation:
                    # # train ---> val (in internal set)
                    # # val --> test (in external set)
                   
                    np.savetxt(f,np.c_[(
                        val_acc[ep_stopped-1], v_val_acc_score,
                        val_auc_score, v_val_auc_score,
                        val_loss[ep_stopped-1], v_val_loss_score, 
                        v_val_mmc_score ,tn, fp, fn, tp,-1,-1, index+1, seed_value)], fmt="%s",delimiter="\t")
                    f.close() 

                 
                
                if options.save_w == 'y' or options.save_entire_w in ['y']: #save weights  
                    if options.model in [ 'svm_model','rf_model','dtc_model','gbc_model','knn_model']:
                        import pickle
                        filename = prefix_models+"s"+str(seed_value)+"ml.sav"
                        pickle.dump(model, open(filename, 'wb'))

                    else:                  
                        model.save_weights(prefix_models+"s"+str(seed_value)+".h5")         
                        model_json = model.to_json()
                        with open(prefix_models+"s"+str(seed_value)+".json", "w") as json_file:
                            json_file.write(model_json)
                    #print("Saved model to disk")
        
            # !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!= finish one fold  !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=
            
        
        if options.type_run in ['vis','visual']: #only visual, not learning
            print  ('mode: VISUALIZATIONS, CREATING IMAGES, NOT LEARNING, skip writing logs.... each run time')

        else: #learning    
            #check point after finishing one running time
            mid_time.append(time.time())   
            if seed_value == options.seed_value_begin:              
                total_t=mid_time[seed_value - options.seed_value_begin ]-start_time
                distant_t=total_t
            else:
                total_t=mid_time[seed_value- options.seed_value_begin ]-start_time
                distant_t=mid_time[seed_value - options.seed_value_begin]-mid_time[seed_value-options.seed_value_begin-1 ]           

            res=(seed_value,np.mean(early_stop_1se),
                np.mean(train_acc_1se_begin),np.mean(train_acc_1se),
                np.mean(val_acc_1se_begin),np.mean(val_acc_1se),
                np.mean(train_loss_1se_begin),np.mean(train_loss_1se),
                np.mean(val_loss_1se_begin),np.mean(val_loss_1se),
                total_t,distant_t)
            temp_all_acc_loss.append(res)       

            if options.save_avg_run in ['y']:
                f=open(prefix_file_log+"_mean_acc.txt",'a')        
                #if options.save_l in ['y'] :
                np.savetxt(f,np.c_[(seed_value,np.mean(early_stop_1se),
                        np.mean(train_acc_1se_begin),np.mean(train_acc_1se),
                        np.mean(val_acc_1se_begin),np.mean(val_acc_1se),
                        np.mean(train_loss_1se_begin),np.mean(train_loss_1se),
                        np.mean(val_loss_1se_begin),np.mean(val_loss_1se),
                        total_t,distant_t)], fmt="%s",delimiter="\t")
                #else:
                #    np.savetxt(f,np.c_[(seed_value,np.mean(early_stop_1se),
                #        np.mean(train_acc_1se_begin),np.mean(train_acc_1se),
                #        np.mean(val_acc_1se_begin),np.mean(val_acc_1se),
                    #  train_loss_1se_begin.mean(),train_loss_1se.mean(),
                    #  val_loss_1se_begin.mean(),val_loss_1se.mean(),
                #        total_t,distant_t)],fmt="%s",delimiter="\t")
                f.close()

            #confusion and auc
            res=(seed_value,np.mean(early_stop_1se),
                    np.mean(train_tn_1se),np.mean(train_fn_1se),
                    np.mean(train_fp_1se),np.mean(train_tp_1se),
                    np.mean(val_tn_1se),np.mean(val_fn_1se),
                    np.mean(val_fp_1se),np.mean(val_tp_1se),
                    (float((np.mean(train_tp_1se)+np.mean(train_tn_1se))/float(np.mean(train_tn_1se)+np.mean(train_fn_1se)+np.mean(train_fp_1se)+np.mean(train_tp_1se)))), #accuracy
                    (float((np.mean(val_tp_1se)+np.mean(val_tn_1se))/float(np.mean(val_tn_1se)+np.mean(val_fn_1se)+np.mean(val_fp_1se)+np.mean(val_tp_1se)))), 
                    float(np.mean(train_auc_1se)) , #auc
                    float(np.mean(val_auc_1se)) ,
                    float(np.mean(val_pre_1se)),
                    float(np.mean(val_recall_1se)),
                    float(np.mean(val_f1score_1se)),
                    float(np.mean(val_mmc_1se)))
            temp_all_auc_confusion_matrix.append(res)    

            if options.save_avg_run in ['y']:
                f=open(prefix_file_log+"_mean_auc.txt",'a')
                np.savetxt(f,np.c_[(seed_value,np.mean(early_stop_1se),
                        np.mean(train_tn_1se),np.mean(train_fn_1se),
                        np.mean(train_fp_1se),np.mean(train_tp_1se),
                        np.mean(val_tn_1se),np.mean(val_fn_1se),
                        np.mean(val_fp_1se),np.mean(val_tp_1se),
                        (float((np.mean(train_tp_1se)+np.mean(train_tn_1se))/float(np.mean(train_tn_1se)+np.mean(train_fn_1se)+np.mean(train_fp_1se)+np.mean(train_tp_1se)))), #accuracy
                        (float((np.mean(val_tp_1se)+np.mean(val_tn_1se))/float(np.mean(val_tn_1se)+np.mean(val_fn_1se)+np.mean(val_fp_1se)+np.mean(val_tp_1se)))), 
                        float(np.mean(train_auc_1se)) , #auc
                        float(np.mean(val_auc_1se)))],fmt="%s",delimiter="\t")
                f.close()
        # !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!= END of all folds of a running time !=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=!=
    
    
    if options.type_run in ['vis','visual']: #only visual, not learning
        print  ('mode: VISUALIZATIONS, CREATING IMAGES, NOT LEARNING, skip writing logs.... skip collecting results')
        print  ('images were created successfully at ' + path_write)

    else: #learning  
        #update to the end of file the mean/sd of all results (acc,loss,...): file2
        
        acc_loss_mean=np.around(np.mean(temp_all_acc_loss, axis=0),decimals=3)
        acc_loss_std=np.around(np.std(temp_all_acc_loss, axis=0),decimals=3)
        if options.save_avg_run in ['y']:
            f=open(prefix_file_log+"_mean_acc.txt",'a')
            np.savetxt(f,acc_loss_mean, fmt="%s",newline="\t")
            np.savetxt(f,acc_loss_std, fmt="%s",newline="\t",header="\n")
            f.close() 
            

        #update to the end of file the mean/sd of all results (auc, tn, tp,...): file3
        
        auc_mean=np.around(np.mean(temp_all_auc_confusion_matrix, axis=0),decimals=3)
        auc_std=np.around(np.std(temp_all_auc_confusion_matrix, axis=0),decimals=3)
        if options.save_avg_run in ['y']:
            f=open(prefix_file_log+"_mean_auc.txt",'a')
            np.savetxt(f,auc_mean, fmt="%s",newline="\t")
            np.savetxt(f,auc_std, fmt="%s",newline="\t",header="\n")
            f.close() 

        finish_time=time.time()
        #save final results to the end of file1
        
        
        f=open(prefix_file_log+"_sum.txt",'a')
        title_cols = np.array([['time','tr_ac',"va_ac","sd","va_au","sd","tn","fn","fp","tp","preci","sd","recall","sd","f1","sd","mmc","sd","epst"]])
        np.savetxt(f,title_cols, fmt="%s",delimiter="\t")   
        TN=auc_mean[6]
        FN=auc_mean[7]
        FP=auc_mean[8]
        TP=auc_mean[9]
        precision=auc_mean[14]
        recall=auc_mean[15]
        f1_score=auc_mean[16]
        mmc=auc_mean[17]

        np.savetxt(f,np.c_[ (np.around( finish_time-start_time,decimals=1),acc_loss_mean[3],acc_loss_mean[5],acc_loss_std[5],auc_mean[13],auc_std[13], 
            TN, FN, FP, TP,precision,auc_std[14],recall,auc_std[15],f1_score,auc_std[16],mmc,auc_std[17],auc_mean[1])],
            fmt="%s",delimiter="\t")
        

        #update name file to acknowledge the running done!
        #print  ('acc='+str(acc_loss_mean[5])
              
        title_cols = np.array([['tr_ac_a','sd_ac',"va_ac_a","sd_ac",'tr_au_a','sd_au',"va_au_a","sd_au",'va_mc_a','sd_mc']])
        np.savetxt(f,title_cols, fmt="%s",delimiter="\t")   
        np.savetxt(f,np.c_[ (
            np.mean(train_acc_all, axis=0),
            np.std(train_acc_all, axis=0),
            np.mean(val_acc_all, axis=0),
            np.std(val_acc_all, axis=0),
            #len(train_acc_all), #in order to check #folds * #run
            #len(val_acc_all),
            np.mean(train_auc_all, axis=0),
            np.std(train_auc_all, axis=0),
            np.mean(val_auc_all, axis=0),
            np.std(val_auc_all, axis=0),
            np.mean(val_mmc_all, axis=0),
            np.std(val_mmc_all, axis=0)
            #len(train_auc_all),
            #len(val_auc_all)
            )] , fmt="%s",delimiter="\t")
        
        f.close()   
        if options.cudaid > -3:    
        #if options.cudaid not in  ["-1",'']:
            #os.rename(res_dir + "/" + prefix+"file_sum.txt", res_dir + "/" + prefix+"gpu"+ str(options.cudaid) + "file_sum_"+str(options.suff_fini)+".txt")  
            os.rename(res_dir + "/" + prefix+"file_sum.txt", res_dir + "/" + prefix+"gpu_file_sum_"+str(options.suff_fini)+".txt")  
        else:
            os.rename(prefix_file_log+"_sum.txt", prefix_file_log+"_sum_"+str(options.suff_fini)+".txt")  
        
        if special_usecase in ['y','yes']: 
            return np.mean(train_acc_all, axis=0), np.mean(val_acc_all, axis=0), np.mean(val_auc_all, axis=0), np.mean(val_mmc_all, axis=0)
        
        print ('mean_val_acc=' + str(acc_loss_mean[5]))
        print  (utils.textcolor_display ('the experiment has done!', type_mes ='inf'))


        return time_text

        
    
