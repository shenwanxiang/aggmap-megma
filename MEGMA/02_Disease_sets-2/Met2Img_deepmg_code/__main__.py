"""
================================================================================
dev_met2img.py : Predicting diseases based on images (EMBEDDINGs) 
        with stratified cross-validation/holdout/predict 
================================================================================
Author: Thanh Hai Nguyen
(**this version is still developping for python3)
created: 25/12/2017' (updated to 30/01/2020)

function : run n times (cross validation or holdout validation) using embeddings 
    with fillup and manifold algorithms (t-SNE, LDA, isomap,...) 
    support color, gray images with manual bins or bins learned from training set
====================================================================================================================
Steps: 
1. read para from the user command line or config or both
    or check the package if --check=y
2. select resources to run (CPU/GPU)
3. select mode of running (predict/learn/vis/config):
    learn: training and testing
    predict: use pretrained models for test (if existing labels) or just for predict (without labels)
    vis: only create images, not learning
    config: conly create config file, not learning nor visualizing
====================================================================================================================
"""

if __name__ == "__main__":  
    #import warnings
    #warnings.simplefilter("ignore")

    #get parameters from command line
    # step 1. read para from the user command line or config or both  #####
    #from deepmg import experiment
    try:
        import experiment
    except ImportError:
        from deepmg import experiment
    options, args = experiment.para_cmd()

    #check whether dependent packages installed correctly
    if options.check in ['y']:
        try: 
            #load all libraries for deepmg
            import configparser
            import os
            import numpy as np
            import random as rn
            import pandas as pd
            import math
            from time import gmtime, strftime
            import time

            import matplotlib as mpl
            mpl.use('Agg')
            from matplotlib import pyplot as plt

            from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score,matthews_corrcoef, confusion_matrix
            from sklearn.model_selection import StratifiedKFold, train_test_split
            from sklearn.manifold import TSNE, LocallyLinearEmbedding, SpectralEmbedding, MDS, Isomap
            from sklearn.cluster import FeatureAgglomeration
            from sklearn.decomposition import PCA, NMF
            from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
            from sklearn import random_projection
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC

            import tensorflow as tf
            import keras as kr
            import keras.callbacks as Callback
            from keras import backend as K, optimizers
            from keras.models import Sequential
            from keras.layers import Activation, Dropout, Flatten, Dense, InputLayer, Conv2D, MaxPooling2D
            from keras.layers import Conv1D, Embedding, MaxPooling1D, GlobalAveragePooling1D
            from keras.applications.resnet50 import ResNet50
            from keras.applications.vgg16 import VGG16
            from keras.applications.inception_resnet_v2 import InceptionResNetV2
            from keras.models import Model
            from keras.layers import LSTM       

            from keras_sequential_ascii import sequential_model_to_ascii_printout
            #import utils_deepmg
            
            
            try:
                import utils_deepmg as utils
            except ImportError:
                from deepmg import utils_deepmg as utils
            
            print (utils.textcolor_display('deepmg can work properly now!', type_mes = 'inf'))
        except ImportError as error:      
            print (error.__class__.__name__ + ": " + utils.textcolor_display(error.message,type_mes = 'er') + '. Please check the installation of dependent modules!!')
        except Exception as exception:
            # Output unexpected Exceptions.
            print(exception, False)
            print(exception.__class__.__name__ + ": " + exception.message)    
        exit()

    #check if read parameters from configuration file
    import os
    if options.config_file != '':
        if os.path.isfile(options.config_file):
            experiment.para_config_file(options)
            #print options
        else:
            print ('config file does not exist!!!')
            exit()
    else:
        experiment.get_default_value(options)

    #convert options which type numeric
    experiment.string_to_numeric(options)
    #check whether parameters all valid
    experiment.validation_para(options)

    # step 2. rselect resources to run (CPU/GPU)  #####   
    # select run either GPU or CPU: we have 4 cases:
    # + <=-3: only use CPU
    # + -2: use cpu if there is no available GPU
    # + -1: use all available GPU
    # + >0: index of GPU for use
    if options.cudaid <= -3 : #run cpu
        print ('you are using cpu')
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""            
        
    else: #use gpu
        if options.cudaid == -2: #select cpu if there is no available gpu 
            try: 
                if options.cudaid > -1: 
                    ##specify idcuda you would like to use            
                    print ('you are using gpu: ' + str(options.cudaid)   ) 
                    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(options.cudaid)        
                else: #==-1
                    #use all available gpu
                    print ('use all available gpu!')

                import tensorflow as tf
                from keras.backend.tensorflow_backend import set_session

                config = tf.ConfigProto()
                if options.gpu_memory_fraction <= 0:
                    config.gpu_options.allow_growth = True
                else:
                    config.gpu_options.per_process_gpu_memory_fraction = options.gpu_memory_fraction   
                set_session(tf.Session(config=config))
            
            except ValueError:            
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
                os.environ["CUDA_VISIBLE_DEVICES"] = ""   
                print ('there is no available GPU for use, so you are running on CPU')
        
        else:
            if options.cudaid > -1: 
                ##specify idcuda you would like to use            
                print ('you are using gpu: ' + str(options.cudaid)    )
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
                os.environ["CUDA_VISIBLE_DEVICES"] = str(options.cudaid)
            
            else: #==-1
                #use all available gpu
                print ('use all available gpu!')

            #set ratio of memory to use, if gpu_memory_fraction <=0: only use necessary memory for running --> recommended!
            import tensorflow as tf
            from keras.backend.tensorflow_backend import set_session
            config = tf.ConfigProto()
            if options.gpu_memory_fraction <=0:
                config.gpu_options.allow_growth = True
            else:
                config.gpu_options.per_process_gpu_memory_fraction = options.gpu_memory_fraction   
            set_session(tf.Session(config=config))


# step 3: select mode of running (predict/learn/vis/config):  #####  
 
    if options.type_run in ['vis','visual']: #if only visualize
        experiment.deepmg_visual(options,args)
        
    elif options.type_run in ['learn','config']:         #if learning
        #print 'test2'
        print (options)
        if options.test_size in [0,1]:   #if use cross-validation
            
            if options.n_folds != 1: #if use k= 1, skip this
                print ('learning with Cross validation')
                time_text = experiment.run_kfold_deepmg(options,args)    
        else: #if set the size of test set, so use holdout validation
            #print 'test3b'   
            time_text = experiment.run_holdout_deepmg(options,args)    
        #print 'test3'      
        
        if options.save_entire_w in ['y'] or options.test_ext in ['y']:        #if get weights on whole dataset  
            if options.n_folds != 1:
                print ('learning with whole training set, then predict on test set')
                experiment.run_holdout_deepmg(options,args, special_usecase = 'train_test_whole',txt_time_pre=time_text)  
            else: #if use k= 1, training on whole training set, and test on test set
                from time import gmtime, strftime
                time_text = str(strftime("%Y%m%d_%H%M%S", gmtime())) 
                experiment.run_holdout_deepmg(options,args, special_usecase = 'train_test_whole',txt_time_pre=time_text)

    elif options.type_run in ['predict']: #if predict or test from a pretrained model
        #print 'test4'      
        experiment.run_holdout_deepmg(options,args, special_usecase = 'predict')   
    
    if options.sound_fini in ['y','yes']:
        import os
        os.system('say "Your experiment has finished. Please collect your results"') 
 