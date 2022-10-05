"""
===================================
MODEL DEFINITION AND SELECTION
===================================
Author: Thanh Hai Nguyen, Team Integromics, ICAN, Paris, France
date: 15/12/2017 (updated to 10/12/2018)
email to contact: nthai@cit.ctu.edu.vn
This module supports building architectures of MLP, LTSM, CNN, VGG-like (baseline), 
    and some funcs for selection models (call_model,grid_search_coef)

Many models are being supported in this module:
1. svm_model: Support Vector Machines
2. rf_model: Random Forests
3. gbc_model: Gradient Boosting
4. model_cnn: Convolutional Neural Networks for images
5. model_pretrained: some famous pretrained nets such as VGG, ResNet50, Incep,...
6. fc_model: Linear Regression, with one fully connected layer
7. model_lstm: LSTM - Long Short Term Memory nets
8. model_mlp: Multilayer Perceptron
9. model_vgglike: simple CNN based on VGG
10. model_cnn1d: Convolutional Neural Networks for 1D data
11. knn_model: KNeighborsClassifier

and others:
11. call_model: select model to learn
12. grid_search_output: Fine-tuned, grid-search for finding the best models
13. cv_run_search: cross-validation for grid-search (called in grid_search_output)
14. grid_search_coef: grid-search to find the best coefficient
15. grid_search_model: Fine-tuned, grid-search for finding the best models (long code)
EarlyStopping_consecutively: for Early Stopping consecutively epoch: which improved the learning compared to Keras early stopping
"""
try:
    from keras import backend as K, optimizers
except Exception as exception:
    from tensorflow.keras import backend as K, optimizers
# https://github.com/keras-team/keras/issues/12379

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, InputLayer, Conv2D, MaxPooling2D
from keras.layers import Conv1D, Embedding, MaxPooling1D, GlobalAveragePooling1D
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model
from keras.layers import LSTM
import keras as kr

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import keras.callbacks as Callback
import keras
from time import gmtime, strftime
import time
from sys import exit

def svm_model (svm_C=1,svm_kernel='linear', num_output = 1):
    """ perform a SVM
    Args:
        C (float): Penalty parameter C of the error term (default python: 1.0)
        kernel (string) : linear, poly, rbf, sigmoid, precomputed (default python: rbf)

    Returns:
        model
    """
    if num_output > 2 :
        return LinearSVC(C=svm_C,kernel=svm_kernel,probability=True)
    else:
        return SVC(C=svm_C,kernel=svm_kernel,probability=True)

def knn_model(knn_n_neighbors=5, knn_algorithm='auto', knn_leaf_size=30, knn_p=2):
    """ perform a KNeighborsClassifier
    Args:
        C (float): Penalty parameter C of the error term (default python: 1.0)
        kernel (string) : linear, poly, rbf, sigmoid, precomputed (default python: rbf)

    Returns:
        model
    """
    return KNeighborsClassifier(n_neighbors=knn_n_neighbors,algorithm=knn_algorithm,leaf_size=knn_leaf_size,p=knn_p)

def dtc_model(rf_max_depth=-1,rf_min_samples_split=2 , rf_random_state = None, rf_max_features=-2):
    if rf_max_features ==-2:
        rf_max_features_v = 'auto'
    elif rf_max_features ==-1:
        rf_max_features_v = None
    else:
        rf_max_features_v = rf_max_features
    if rf_max_depth == -1:
        return DecisionTreeClassifier(min_samples_split=rf_min_samples_split, max_features=rf_max_features, random_state=rf_random_states)
    else:
        return DecisionTreeClassifier(max_depth=rf_max_depth, min_samples_split=rf_min_samples_split, max_features=rf_max_features, random_state=rf_random_states)

def rf_model (rf_n_estimators=500,rf_max_depth=-1,rf_min_samples_split=2 , rf_random_state = None, rf_max_features=-2):
    """ Random Forest
    Args:
        rf_n_estimators (int): number of tree (default python n_estimators=10)
        rf_max_depth: The maximum depth of the tree (default=None)
        min_samples_split: The minimum number of samples required to split an internal node  (default=2)
        rf_max_depth: The maximum depth of the tree. 
            If -1, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        rf_random_state If int, random_state is the seed used by the random number generator;
    Returns:
        model
    """
    if rf_max_features ==-2:
        rf_max_features_v = 'auto'
    elif rf_max_features ==-1:
        rf_max_features_v = None
    else:
        rf_max_features_v = rf_max_features
    if rf_max_depth == -1:
        return RandomForestClassifier(n_estimators=rf_n_estimators, min_samples_split=rf_min_samples_split, random_state= rf_random_state, max_features=rf_max_features_v)
    else:
        return RandomForestClassifier(n_estimators=rf_n_estimators, min_samples_split=rf_min_samples_split,max_depth=rf_max_depth, random_state=rf_random_state,max_features=rf_max_features_v)

def gbc_model (rf_n_estimators=500,rf_max_depth=-1,rf_min_samples_split=2, rf_random_state = None, rf_max_features=-2):
    """ GradientBoosting
    Args:
        rf_n_estimators (int): number of tree (default python n_estimators=10)
        rf_max_depth: The maximum depth of the tree (default=None)
        min_samples_split: The minimum number of samples required to split an internal node  (default=2)
        rf_max_depth: if -1, use default = 3

    Returns:
        model
    """
    if rf_max_features ==-2:
        rf_max_features_v = 'auto'
    elif rf_max_features ==-1:
        rf_max_features_v = None
    else:
        rf_max_features_v = rf_max_features
    
    if rf_max_depth == -1:
        return GradientBoostingClassifier(n_estimators=rf_n_estimators,  min_samples_split=rf_min_samples_split, random_state = rf_random_state, max_features=rf_max_features_v)
    else:
        return GradientBoostingClassifier(n_estimators=rf_n_estimators,  max_depth=rf_max_depth, min_samples_split=rf_min_samples_split,random_state = rf_random_state, max_features=rf_max_features_v)
    

def fc_model_log(input_reshape=(32,32,3),num_classes=2, optimizers_func='adam', lr_rate=-1, lr_decay=0, 
    loss_func='binary_crossentropy', dropout_fc=0) :
    """ architecture with only one fully connected layer
    Args:
        input_reshape (array): dimension of input
        num_classes (int): the number of output of the network
        optimizers_func (string): optimizers function
        lr_rate (float): learning rate, if use -1 then use default values of the optimizer
        lr_decay (float): learning rate decay
        loss_func (string): loss function

    Returns:
        model
    """
    model = Sequential()
    model.add(InputLayer(input_shape=input_reshape))  
    model.add(Flatten())    

    print(' dropout ' + str(dropout_fc))
    if dropout_fc > 0:
        model.add(Dropout(dropout_fc))

    if num_classes==1:
        model.add(Dense(1, activation='sigmoid'))
    else:
       # model.add(Dense(num_classes, activation='softmax'))
        model.add(Dense(num_classes, nonlinearity='log_softmax'))
        #DenseLayer(layer, num_units=10, nonlinearity=log_softmax)
    
    if lr_rate > -1:           
        if optimizers_func=="adam":
            kr.optimizers.Adam(lr=lr_rate, decay=lr_decay)
        elif optimizers_func=="sgd":
            kr.optimizers.SGD(lr=lr_rate, decay=lr_decay)  
         
    #mean_absolute_error, squared_hinge, hinge
    #t2d: aldenla: mean_squared_error: 0.929909176323/0.584033613701, binary_crossentropy: 0.903746737655/0.598487395384
    model.compile(
            loss=loss_func,
            #  loss='mean_squared_error', #0.929909176323/0.584033613701 (10cv)            
            #  loss='categorical_crossentropy', #0.903746737655/0.598487395384
            #  loss='mean_squared_logarithmic_error', #0.92893934648/0.586890756558
            #  loss='squared_hinge', #0.929909176323/0.584033613701, the same mean_squared_error
            #  loss='hinge', #did not work
            #  loss='categorical_crossentropy',
            optimizer=optimizers_func,
            # optimizer='rmsprop',
            metrics=['accuracy'])
         
    return model

def model_cnn(input_reshape=(32,32,3),num_classes=2,optimizers_func='Adam',numfilter=20, 
        filtersize=3,numlayercnn_per_maxpool=1,nummaxpool=1,maxpoolsize=2,
        dropout_cnn=0, dropout_fc=0,lr_rate=0.0005,lr_decay=0, loss_func='binary_crossentropy', padded='n' ) :
    """ architecture CNNs with specific filters, pooling...
    Args:
        input_reshape (array): dimension of input
        num_classes (int): the number of output of the network
        optimizers_func (string): optimizers function
        lr_rate (float): learning rate, if use -1 then use default values of the optimizer
        lr_decay (float): learning rate decay
        loss_func (string): loss function

        numfilter (int): number of filters (kernels) for each cnn layer
        filtersize (int): filter size
        numlayercnn_per_maxpool (int): the number of convolutional layers before each max pooling
        nummaxpool (int): the number of max pooling layer
        maxpoolsize (int): pooling size
        dropout_cnn (float): dropout at each cnn layer
        dropout_fc (float): dropout at the FC (fully connected) layers
        padded: padding 'same' to input to keep the same size after 1st conv layer
        
    Returns:
        model
    """
    model = Sequential()
    model.add(InputLayer(input_shape=input_reshape))  

    for j_pool in range(1,nummaxpool+1):
        print(("j_pool"+str(j_pool)))
        for i_layer in range(1,numlayercnn_per_maxpool+1):
            if i_layer==1:
                if padded=='y':
                    #use padding
                    model.add(Conv2D(numfilter, (filtersize, filtersize), padding='same'))
                else:
                    #do not use padding
                    model.add(Conv2D(numfilter, (filtersize, filtersize)))
            else:                
                model.add(Conv2D(numfilter, (filtersize, filtersize)))
            model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(maxpoolsize,maxpoolsize)))
        if dropout_cnn > 0:
            model.add(Dropout(dropout_cnn))
        #model.add(Conv2D(1, (1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    if dropout_fc > 0:
        model.add(Dropout(dropout_fc))
    
    # model.add(Dense(num_classes))
    # model.add(Activation('softmax'))   
    if num_classes==1:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))
       

    if lr_rate > -1:           
        if optimizers_func=="adam":
            optimizers_func = kr.optimizers.Adam(lr=lr_rate, decay=lr_decay)
        elif optimizers_func=="sgd":
            optimizers_func = kr.optimizers.SGD(lr=lr_rate, decay=lr_decay)  
        elif optimizers_func=="rmsprop":
            optimizers_func = kr.optimizers.rmsprop(lr=lr_rate, decay=lr_decay) 
  

    model.compile(loss=loss_func,
            optimizer=optimizers_func,
            metrics=['accuracy'])
        
    return model

def model_pretrained (input_reshape,name_model ='vgg16',
    num_classes=2,optimizers_func='Adam', loss_func='binary_crossentropy', pretrained_file = ''):
    #input_tensor = Input(shape=(train_x.shape[1],train_x.shape[2], options.channel)) 
    if name_model == 'resnet50':
        model = ResNet50(weights='imagenet',input_tensor=input_reshape)
    elif name_model == 'vgg16':
        model = VGG16(weights='imagenet',input_tensor=input_reshape)
    elif name_model == 'inceptionresnetV2':
        model = InceptionResNetV2(weights='imagenet',input_tensor=input_reshape)
    elif name_model == 'pretrained':

        from keras.models import model_from_json
        import os
        # if use pretrained file you have
        if pretrained_file[len(pretrained_file)-2]+pretrained_file[len(pretrained_file)-1] in ['ml']:
            #if use machine learning pretrained
            pretrained_file_ml = pretrained_file + str(".sav")
            if os.path.isfile(pretrained_file_ml):
                import pickle
                model = pickle.load(open(pretrained_file_ml, 'rb'))
                return model
            else:
                print('the pre-trained (sav) file you provided does not exists!! (Note: do not add extension (e.g .sav) of file)')
                print(pretrained_file_ml)
                exit()

        else:
            file_mcnn = pretrained_file + str(".json")
            file_wcnn = pretrained_file + str(".h5")
            if os.path.isfile(file_mcnn):
                json_file = open(file_mcnn, 'r')
                loaded_model_jsoncnn = json_file.read()
                json_file.close()
                model = model_from_json(loaded_model_jsoncnn)
                model.load_weights(file_wcnn)
                model.compile(optimizer=optimizers_func, loss=loss_func, metrics=['accuracy'])
                return model
            else:
                print('the pre-trained (json) file you provided does not exists!! (Note: do not add extension (e.g .json) of file)')
                exit()
    else:
        print('this pre-trained model is not supported!!')
        exit()
        
    
    x = model.output
    #x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    #x = Dense(1024, activation='relu')(x)
    if num_classes <= 1:
        predictions = Dense(1, activation='sigmoid')(x)
    else:
        predictions = Dense(num_classes, activation='softmax')(x)
        
    #model = Model(inputs=(85,3,20,20),outputs=predictions) 
    model = Model(inputs=model.input,outputs=predictions)                
    model.compile(optimizer=optimizers_func, loss=loss_func, metrics=['accuracy'])

    return model
    
def fc_model(input_reshape=(32,32,3),num_classes=2, optimizers_func='adam', lr_rate=-1, lr_decay=0, 
    loss_func='binary_crossentropy', dropout_fc=0) :
    """ architecture with only one fully connected layer
    Args:
        input_reshape (array): dimension of input
        num_classes (int): the number of output of the network
        optimizers_func (string): optimizers function
        lr_rate (float): learning rate, if use -1 then use default values of the optimizer (default: lr=0.001-->adam)
        lr_decay (float): learning rate decay
        loss_func (string): loss function

    Returns:
        model
    """
    model = Sequential()
    model.add(InputLayer(input_shape=input_reshape))  
    model.add(Flatten())    

    print(' dropout ' + str(dropout_fc))
    if dropout_fc > 0:
        model.add(Dropout(dropout_fc))

    if num_classes==1:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))
    
    if lr_rate > -1:           
        if optimizers_func=="adam":
            optimizers_func = kr.optimizers.Adam(lr=lr_rate, decay=lr_decay)
        elif optimizers_func=="sgd":
            optimizers_func = kr.optimizers.SGD(lr=lr_rate, decay=lr_decay)  
        elif optimizers_func=="rmsprop":
            optimizers_func = kr.optimizers.rmsprop(lr=lr_rate, decay=lr_decay) 
         
    #mean_absolute_error, squared_hinge, hinge
    #t2d: aldenla: mean_squared_error: 0.929909176323/0.584033613701, binary_crossentropy: 0.903746737655/0.598487395384
    model.compile(
            loss=loss_func,
            #  loss='mean_squared_error', #0.929909176323/0.584033613701 (10cv)            
            #  loss='categorical_crossentropy', #0.903746737655/0.598487395384
            #  loss='mean_squared_logarithmic_error', #0.92893934648/0.586890756558
            #  loss='squared_hinge', #0.929909176323/0.584033613701, the same mean_squared_error
            #  loss='hinge', #did not work
            #  loss='categorical_crossentropy',
            optimizer=optimizers_func,
            # optimizer='rmsprop',
            metrics=['accuracy'])
         
    return model

def model_lstm(input_reshape=(1,381),num_classes=1, optimizers_func='adam', lr_rate=-1, lr_decay=0, loss_func='mae', num_neurons=100,numlayer_layer=1) :
    print('model_lstm:'+str( input_reshape))
    model = Sequential()
    model.add(LSTM(num_neurons, input_shape=input_reshape))
    for i_layer in range(1,numlayer_layer):
        model.add(LSTM(num_neurons))  

    
  #  model.compile(loss='mae', optimizer='adam')
    if num_classes==1:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))
   
    
    if lr_rate > -1:           
        if optimizers_func=="adam":
            optimizers_func = kr.optimizers.Adam(lr=lr_rate, decay=lr_decay)
        elif optimizers_func=="sgd":
            optimizers_func = kr.optimizers.SGD(lr=lr_rate, decay=lr_decay)  
        elif optimizers_func=="rmsprop":
            optimizers_func = kr.optimizers.rmsprop(lr=lr_rate, decay=lr_decay) 

    #mean_absolute_error, squared_hinge, hinge
    #t2d: aldenla: mean_squared_error: 0.929909176323/0.584033613701, binary_crossentropy: 0.903746737655/0.598487395384
    model.compile(
            loss=loss_func,          
            optimizer=optimizers_func,
            # optimizer='rmsprop',
            metrics=['accuracy'])
         
    return model

def model_mlp(input_reshape=(32,32,3),num_classes=2,optimizers_func='adam',
            num_neurons=500, numlayer_layer=1, dropout_fc=0, lr_rate=0.0005,
            lr_decay=0, loss_func='binary_crossentropy' ):
    """ architecture MLP
    Args:
        input_reshape (array): dimension of input
        num_classes (int): the number of output of the network
        optimizers_func (string): optimizers function
        lr_rate (float): learning rate, if use -1 then use default values of the optimizer
        lr_decay (float): learning rate decay
        loss_func (string): loss function

        num_neurons (int): number of neurons
        numlayer_layer: number of percepton layers
        dropout_fc: dropout at each layer       
        
    Returns:
        model with architecture MLP (multi-layer perception)
    """
    model = Sequential()
    model.add(InputLayer(input_shape=input_reshape))  

    model.add(Flatten())
    for i_layer in range(1,numlayer_layer+1):
        if i_layer==1:
            model.add(Dense(num_neurons))
        else:
            if dropout_fc > 0:
                model.add(Dropout(dropout_fc))
  
           
            model.add(Dense(num_neurons))      
    if dropout_fc > 0:
        model.add(Dropout(dropout_fc))  
   
    if num_classes==1:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))        

    if lr_rate > -1:           
        if optimizers_func=="adam":
            optimizers_func = kr.optimizers.Adam(lr=lr_rate, decay=lr_decay)
        elif optimizers_func=="sgd":
            optimizers_func = kr.optimizers.SGD(lr=lr_rate, decay=lr_decay)  
        elif optimizers_func=="rmsprop":
            optimizers_func = kr.optimizers.rmsprop(lr=lr_rate, decay=lr_decay) 

    model.compile(loss=loss_func,
            optimizer=optimizers_func,
            metrics=['accuracy'])        
    return model

def model_cnn4_dropout(input_reshape,num_classes, optimizers_func=optimizers.SGD(lr=0.0005, momentum=0.1, decay=1e-6)) :
    model = Sequential()
    model.add(InputLayer(input_shape=input_reshape))  
    model.add(Conv2D(16, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer=optimizers_func,
            # optimizer='rmsprop',
            metrics=['accuracy'])
           # metrics=['accuracy','auc_epoch_func'])
    return model

def model_vgglike(input_reshape,num_classes=2, optimizers_func='adam', 
    lr_rate=-1, lr_decay=0, loss_func='binary_crossentropy',
    padded='y',dropout_fc=0, dropout_cnn=0):
    """ model VGG-based with basic parameters, used as a baseline
    Args:
        input_reshape (array): dimension of input
        num_classes (int): number of output of the network
               
    Returns:
        model VGG
    """
    model = Sequential()
    # input: DxD images with 3 channels -> (100, 100, 3) tensors.
    # # this applies 32 convolution filters of size 3x3 each.
    if padded=='y':
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_reshape,padding='same'))
    else:
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_reshape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout_cnn > 0:
        model.add(Dropout(dropout_cnn))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout_cnn > 0:
        model.add(Dropout(dropout_cnn))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    if dropout_fc > 0:
        model.add(Dropout(dropout_fc))
   
    if num_classes==1:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))
    
    if lr_rate > -1:           
        if optimizers_func=="adam":
            optimizers_func = kr.optimizers.Adam(lr=lr_rate, decay=lr_decay)
        elif optimizers_func=="sgd":
            optimizers_func = kr.optimizers.SGD(lr=lr_rate, decay=lr_decay)  
        elif optimizers_func=="rmsprop":
            optimizers_func = kr.optimizers.rmsprop(lr=lr_rate, decay=lr_decay) 
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=loss_func, optimizer=optimizers_func, metrics=['accuracy'])
    return model

def model_cnn1d(input_reshape,num_classes=1, optimizers_func ='rmsprop', 
        numfilter=20, filtersize=3,numlayercnn_per_maxpool=1,nummaxpool=1,maxpoolsize=2,
        dropout_cnn=0, dropout_fc=0,lr_rate=0.0005,lr_decay=0, loss_func='binary_crossentropy',padded='n' ):
   
    """ model CNN1D for 1D data
    Args:
        input_reshape (array): dimension of input
        num_classes (int): number of output of the network
        numfilter (int): number of filters/kernels
        filtersize (int): size of filter
        numlayercnn_per_maxpool (int): number of conv layer stacked before one max pooling
        nummaxpool (int): number of max pooling
        maxpoolsize (int): size of max pooling
        dropout_cnn (float): dropout rate at each conv
        dropout_fc (float): dropout rate FC    
        padded: padding 'same' to input to keep the same size after 1st conv layer
    Returns:
        model 
    """

    model = Sequential()
    print('model_cnn1d.. '+ str(input_reshape))
    model.add(InputLayer(input_shape=input_reshape))  
   
    for j_pool in range(1,nummaxpool+1):
        print(("j_pool"+str(j_pool)))
        for i_layer in range(1,numlayercnn_per_maxpool+1):                
            if dropout_cnn > 0:
                model.add(Dropout(dropout_cnn))    
            if i_layer==1:
                if padded=='y':
                    model.add(Conv1D(numfilter, filtersize, activation='relu',padding='same')) 
                else:    
                    model.add(Conv1D(numfilter, filtersize, activation='relu')) 
        if j_pool != nummaxpool:
            model.add(MaxPooling1D(maxpoolsize))
    
    #model.add(GlobalAveragePooling1D())
    model.add(Flatten())
    if dropout_fc > 0:
        model.add(Dropout(dropout_fc))
  
     
    if lr_rate > -1:           
        if optimizers_func=="adam":
            optimizers_func = kr.optimizers.Adam(lr=lr_rate, decay=lr_decay)
        elif optimizers_func=="sgd":
            optimizers_func = kr.optimizers.SGD(lr=lr_rate, decay=lr_decay)  
        elif optimizers_func=="rmsprop":
            optimizers_func = kr.optimizers.rmsprop(lr=lr_rate, decay=lr_decay) 
    
    print('optimizers_func=========')
    print(optimizers_func)
    print('learning_rate=========')
    print(lr_rate)
   # model.add(Flatten())
    if num_classes==1:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))   

    model.compile(loss=loss_func,
                optimizer=optimizers_func,
                metrics=['accuracy'])
    return model

def model_con1d_baseline(input_reshape,optimizers_func='rmsprop', loss_func='binary_crossentropy',num_classes=1):
    model = Sequential()
    print(input_reshape)
    model.add(InputLayer(input_shape=input_reshape))  
    
    model.add(Conv1D(64, 3, activation='relu'))
   # model.add(Conv2D(numfilter, (filtersize, filtersize), padding='same'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.5))
   # model.add(Flatten())
    if num_classes==1:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))   

    model.compile(loss=loss_func,
                optimizer=optimizers_func,
                metrics=['accuracy'])
    return model

def model_lstm_baseline(input_reshape,num_classes=1, optimizers_func='rmsprop', loss_func='binary_crossentropy'):
    model = Sequential()
    print(input_reshape)
    model.add(LSTM(16,input_shape=input_reshape))
   
   # model.add(LSTM(32, stateful=True))
    if num_classes==1:
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(num_classes, activation='softmax'))   

    model.compile(loss=loss_func,
                optimizer=optimizers_func, # rmsprop
                metrics=['accuracy'])
    return model
    
def grid_search_coef(X, y, 
                g_type_model, g_input_shape, g_num_classes, g_optimizer,
                g_learning_rate, g_learning_decay, g_loss_func,
                g_ml_number_filters, g_ml_numlayercnn_per_maxpool, g_ml_dropout_rate_fc, 
                g_mc_nummaxpool, g_mc_poolsize, g_mc_dropout_rate_cnn, g_mc_filtersize,
                g_batch_size, g_epochs_num,g_mc_padding, time_loop,
                early_stopping,
                coef_test_ini = 512,  num_grid_coef = 5, cv_time = 5, seed_v = 1, debug = 0,
                save_log_file = '0'
                ):
    """ choose best coefficient for data using grid search
    Args:
        X: input
        y: labels of input
        g_type_model (string): type of model
        g_input_shape (array): dimension of input
        g_num_classes (int): number of output of the network
        g_optimizer (string): optimizer function
        g_learning_rate (float): learning rate
        g_learning_decay (float): learning rate decay
        g_loss_func (string): loss function
        g_ml_number_filters (int): number of filters
        g_ml_numlayercnn_per_maxpool (int): number of cnn layer before each pooling
        g_ml_dropout_rate_fc (float): dropout at FC
        g_mc_nummaxpool (int): number of pooling layers
        g_mc_poolsize (int): pooling size
        g_mc_dropout_rate_cnn (float): dropout at CNN layers
        g_mc_filtersize (int): filter size  
        g_batch_size (int): mini batch size
        g_epochs_num (int): #epoch
        coef_test_ini (float): coefficient initialized at the beginning
        num_grid_coef (int): the number of coefficient to run grid search
        cv_time (int): cross validation for each coef
        seed_v (int): seed to shuffle data
        debug (int)
        save_log_file (string) : path to log to write
        time_loop (int): the time of loop

    Returns:
        the values of coefs used, the best coef; acc, auc of coefs
    """
    coef_test = []    
    all_acc_grid= []
    all_auc_grid=[]   
    all_acc_train_grid= []
    all_ep_estop_train_grid= []
   
    train_x1 = []
    max_temp=0
    index_max=0    
    max_auc_temp=0

    #if save results to log
    if save_log_file != '0':
        f=open(save_log_file,'a')
        title_cols = np.array([["time",'coef','seed','ep_est','tr_ac','val_a','val_au']])
        np.savetxt(f,title_cols, fmt="%s",delimiter="\t")
        f.close()

    #try #num_grid_coef coefficient(s)
    for i in range(0,num_grid_coef):
    
        print('time: ' +str(i))
        #each next loop will compute with value = value * 2
        if i == 0: # for the first time, use ini_value
            coef_test.append( coef_test_ini)
        else:
            coef_test.append( coef_test [i-1] * 2)

        #divide X with coef
        train_x1 = X / float (coef_test [i])    
        train_y1= y
        
        #reset index to range into folds
        train_y1=train_y1.reset_index()     
        train_y1=train_y1.drop('index', 1)
       
        acc_grid= []
        auc_grid=[]   
        acc_train_grid= []
        ep_estop_train_grid= []
        
        for seed_i in range(seed_v,seed_v+time_loop) :
            acc_grid_cv = []
            auc_grid_cv = []
            acc_train_grid_cv = []
            ep_estop_train_grid_cv = []
            skf_grid=StratifiedKFold(n_splits=cv_time, random_state=seed_i, shuffle= True)
            for index_gr, (train_indices_gr,val_indices_gr) in enumerate(skf_grid.split(train_x1, train_y1.x)):
                print("Training on fold " + str(index_gr+1) + "/"+str(cv_time)+"...")
            
                train_x_grid = []
                train_y_grid = []
                val_x_grid = []
                val_y_grid = []           
            
                train_x_grid, val_x_grid = train_x1 [train_indices_gr], train_x1[val_indices_gr]
                train_y_grid, val_y_grid = train_y1.x [train_indices_gr], train_y1.x[val_indices_gr]

                if debug > 0:
                    print(train_y)
                    print(val_y_grid)
                
                if g_num_classes == 2:
                    train_y_grid=kr.utils.np_utils.to_categorical(train_y_grid)
                    val_y_grid = kr.utils.np_utils.to_categorical(val_y_grid)
                
                #call model
                model1 = call_model (type_model=g_type_model, 
                    m_input_shape = g_input_shape, m_num_classes= g_num_classes, m_optimizer = g_optimizer,
                    m_learning_rate = g_learning_rate, m_learning_decay=g_learning_decay, m_loss_func=g_loss_func,
                    ml_number_filters=g_ml_number_filters,ml_numlayercnn_per_maxpool=g_ml_numlayercnn_per_maxpool, ml_dropout_rate_fc=g_ml_dropout_rate_fc, 
                    mc_nummaxpool=g_mc_nummaxpool, mc_poolsize= g_mc_poolsize, mc_dropout_rate_cnn=g_mc_dropout_rate_cnn, mc_filtersize=g_mc_filtersize, mc_padding=g_mc_padding)
                
                print('coef tune: dim=')
                print(train_x_grid.shape)
                print(val_x_grid.shape)
                
                history1 = model1.fit(train_x_grid, train_y_grid, 
                        epochs = g_epochs_num, 
                        batch_size=g_batch_size, 
                        validation_data=(val_x_grid, val_y_grid),
                        callbacks=[early_stopping],
                        shuffle=False)  

                Y_pred = model1.predict(val_x_grid, verbose=2)
                val_auc_score=roc_auc_score(val_y_grid, Y_pred)
                val_acc_ = history1.history['val_acc']
                train_acc_ = history1.history['acc']
                ep_stopped = len(val_acc_)

                ep_estop_train_grid_cv .append(ep_stopped)
                acc_train_grid_cv .append (train_acc_[ep_stopped-1])
                acc_grid_cv .append(val_acc_ [ep_stopped-1])   
                auc_grid_cv .append(val_auc_score)     

            #mean of k-cross-validation (1 time of run)
            mean_acc_cv = np.around( np.mean(acc_grid_cv,axis=0), decimals=5)
            mean_auc_cv = np.around( np.mean(auc_grid_cv,axis=0), decimals=5)
            mean_ep_estop_train_grid_cv = np.around( np.mean(ep_estop_train_grid_cv,axis=0), decimals=5)
            mean_acc_train_grid_cv = np.around( np.mean(acc_train_grid_cv,axis=0), decimals=5)
        
            acc_grid.append (mean_acc_cv)  
            auc_grid.append (mean_auc_cv)    
            ep_estop_train_grid.append(mean_ep_estop_train_grid_cv)        
            acc_train_grid.append (mean_acc_train_grid_cv)  

            if save_log_file != '0':
                f=open(save_log_file,'a')                  
                np.savetxt(f,np.c_[(i+1,coef_test[i],seed_i,mean_ep_estop_train_grid_cv,mean_acc_train_grid_cv,mean_acc_cv,mean_auc_cv)], fmt="%s",delimiter="\t")                 
                f.close()
                 

        #mean of time_loop time of runs of an coef
        mean_acc = np.around( np.mean(acc_grid,axis=0), decimals=5)
        mean_auc = np.around( np.mean(auc_grid,axis=0), decimals=5)
        mean_ep_estop_train = np.around( np.mean(ep_estop_train_grid,axis=0), decimals=5)
        mean_acc_train = np.around( np.mean(acc_train_grid,axis=0), decimals=5)

        all_acc_grid.append (mean_acc)
        all_auc_grid.append (mean_auc)
        all_acc_train_grid.append (mean_acc_train)
        all_ep_estop_train_grid.append (mean_ep_estop_train )

        #compare to get the best
        if  mean_acc  > max_temp or (mean_auc == max_temp and mean_auc > max_auc_temp) :
            max_temp =  mean_acc
            index_max = i
            max_auc_temp = mean_auc
    return coef_test, all_acc_grid, all_auc_grid, index_max, all_acc_train_grid , all_ep_estop_train_grid         

def grid_search_model(X, y, 
                g_type_model, g_input_shape, g_num_classes, g_optimizer,
                g_learning_rate, g_learning_decay, g_loss_func,
                g_ml_dropout_rate_fc, 
                g_mc_nummaxpool, g_mc_poolsize, g_mc_dropout_rate_cnn, g_mc_filtersize,
                g_batch_size, g_epochs_num,g_mc_padding, time_loop,
                early_stopping,
                layers_list, neuron_filter_list,
                cv_time = 5, seed_v = 1, debug = 0,
                save_log_file = '0', visualize_model=0, time_b=time.time()
                ):
    """ choose best model for data using grid search
    Args:
       

    Returns:
        the results from different models
    """
    layers= [] 
    filters= []     
    all_acc_grid= []
    all_auc_grid=[]   
    all_acc_train_grid= []
    all_ep_estop_train_grid= []
   
    train_x1 = []
    max_temp=0
    index_max=0    
    max_auc_temp=0

    #if save results to log
    if save_log_file != '0':
        f=open(save_log_file,'a')
        title_cols = np.array([["time",'layers','filters','seed','ep_est','tr_ac','val_a','val_au',"1time","alltime"]])
        np.savetxt(f,title_cols, fmt="%s",delimiter="\t")
        f.close()

    #try #num_grid_coef coefficient(s)
    for l_i in range(0,len(layers_list)):
    
        print('#####layer=' +str(layers_list [l_i]))
        
        for k_i in range(0,len(neuron_filter_list)):
            print('###############layer=' +str(layers_list [l_i]) + '##########filters/neurons=' + str(neuron_filter_list [k_i]))
            #get #layers and #filter/neurons for model
            g_ml_numlayercnn_per_maxpool = layers_list [l_i]
            g_ml_number_filters = neuron_filter_list [k_i]

            layers.append(layers_list [l_i]) 
            filters.append(neuron_filter_list[k_i])
            #get train/val
            train_x1 = X     
            train_y1= y
            
            #reset index to range into folds
            train_y1=train_y1.reset_index()     
            train_y1=train_y1.drop('index', 1)
        
            acc_grid= []
            auc_grid=[]   
            acc_train_grid= []
            ep_estop_train_grid= []
            
            #measure time
            time_begin = time.time()

            #the training loops here with #time_loop
            for seed_i in range(seed_v,seed_v+time_loop) :
                acc_grid_cv = []
                auc_grid_cv = []
                acc_train_grid_cv = []
                ep_estop_train_grid_cv = []
                #set k-fold
                skf_grid=StratifiedKFold(n_splits=cv_time, random_state=seed_i, shuffle= True)
                for index_gr, (train_indices_gr,val_indices_gr) in enumerate(skf_grid.split(train_x1, train_y1.x)):
                    print("Training on fold " + str(index_gr+1) + "/"+str(cv_time)+"...")
                
                    train_x_grid = []
                    train_y_grid = []
                    val_x_grid = []
                    val_y_grid = []           
                
                    train_x_grid, val_x_grid = train_x1 [train_indices_gr], train_x1[val_indices_gr]
                    train_y_grid, val_y_grid = train_y1.x [train_indices_gr], train_y1.x[val_indices_gr]

                    if debug > 0:
                        print(train_y)
                        print(val_y_grid)
                    
                    if g_num_classes == 2:
                        train_y_grid=kr.utils.np_utils.to_categorical(train_y_grid)
                        val_y_grid = kr.utils.np_utils.to_categorical(val_y_grid)
                    
                    #call model
                    model1 = call_model (type_model=g_type_model, 
                        m_input_shape = g_input_shape, m_num_classes= g_num_classes, m_optimizer = g_optimizer,
                        m_learning_rate = g_learning_rate, m_learning_decay=g_learning_decay, m_loss_func=g_loss_func,
                        ml_number_filters=g_ml_number_filters,ml_numlayercnn_per_maxpool=g_ml_numlayercnn_per_maxpool, ml_dropout_rate_fc=g_ml_dropout_rate_fc, 
                        mc_nummaxpool=g_mc_nummaxpool, mc_poolsize= g_mc_poolsize, mc_dropout_rate_cnn=g_mc_dropout_rate_cnn, mc_filtersize=g_mc_filtersize, mc_padding=g_mc_padding)
                    
                    if visualize_model > 0:              #visualize architecture of model  
                        from keras_sequential_ascii import sequential_model_to_ascii_printout
                        sequential_model_to_ascii_printout(model1)     
                    print('dim of train/val=')
                    print(train_x_grid.shape)
                    print(val_x_grid.shape)
                    
                    history1 = model1.fit(train_x_grid, train_y_grid, 
                            epochs = g_epochs_num, 
                            batch_size=g_batch_size, 
                            validation_data=(val_x_grid, val_y_grid),
                            callbacks=[early_stopping],
                            shuffle=False)  
                    #compute auc,acc,epoch stopped
                    Y_pred = model1.predict(val_x_grid, verbose=2)
                    val_auc_score=roc_auc_score(val_y_grid, Y_pred)
                    val_acc_ = history1.history['val_acc']
                    train_acc_ = history1.history['acc']
                    ep_stopped = len(val_acc_)
                    #save auc,acc,epoch stopped to array
                    ep_estop_train_grid_cv .append(ep_stopped)
                    acc_train_grid_cv .append (train_acc_[ep_stopped-1])
                    acc_grid_cv .append(val_acc_ [ep_stopped-1])   
                    auc_grid_cv .append(val_auc_score)     

                #mean of k-cross-validation (1 time of run)
                mean_acc_cv = np.around( np.mean(acc_grid_cv,axis=0), decimals=5)
                mean_auc_cv = np.around( np.mean(auc_grid_cv,axis=0), decimals=5)
                mean_ep_estop_train_grid_cv = np.around( np.mean(ep_estop_train_grid_cv,axis=0), decimals=5)
                mean_acc_train_grid_cv = np.around( np.mean(acc_train_grid_cv,axis=0), decimals=5)
            
                acc_grid.append (mean_acc_cv)  
                auc_grid.append (mean_auc_cv)    
                ep_estop_train_grid.append (mean_ep_estop_train_grid_cv)        
                acc_train_grid.append (mean_acc_train_grid_cv)  

                if save_log_file != '0':
                    f=open(save_log_file,'a')        
                    print('time$$$$$$$$$$ ====' +str(l_i * len(neuron_filter_list) + k_i))
                   
                    np.savetxt(f,np.c_[(l_i * len(neuron_filter_list) + k_i,layers_list[l_i],
                        neuron_filter_list [k_i],seed_i,mean_ep_estop_train_grid_cv,
                        mean_acc_train_grid_cv,mean_acc_cv,mean_auc_cv, np.around( time.time()- time_begin , decimals=2),  np.around( time.time() - time_b , decimals=2))], fmt="%s",delimiter="\t")                 
                    f.close()
                    

            #mean of time_loop time of runs of an coef
            mean_acc = np.around( np.mean(acc_grid,axis=0), decimals=5)
            mean_auc = np.around( np.mean(auc_grid,axis=0), decimals=5)
            mean_ep_estop_train = np.around( np.mean(ep_estop_train_grid,axis=0), decimals=5)
            mean_acc_train = np.around( np.mean(acc_train_grid,axis=0), decimals=5)

            all_acc_grid.append (mean_acc)
            all_auc_grid.append (mean_auc)
            all_acc_train_grid.append (mean_acc_train)
            all_ep_estop_train_grid.append (mean_ep_estop_train )

            #compare to get the best
            if  mean_acc  > max_temp or (mean_auc == max_temp and mean_auc > max_auc_temp) :
                max_temp =  mean_acc
                index_max = l_i * len(neuron_filter_list) + k_i
                max_auc_temp = mean_auc
    return layers, filters, all_acc_grid, all_auc_grid, index_max, all_acc_train_grid , all_ep_estop_train_grid         

def grid_search_output(X, y, 
                g_type_model, g_input_shape, g_num_classes, g_optimizer,
                g_learning_rate, g_learning_decay, g_loss_func,
                g_ml_number_filters, g_ml_numlayercnn_per_maxpool, g_ml_dropout_rate_fc, 
                g_mc_nummaxpool, g_mc_poolsize, g_mc_dropout_rate_cnn, g_mc_filtersize,
                g_batch_size, g_epochs_num,g_mc_padding,
            coef_test_ini = 512,  num_grid_coef = 5, cv_time = 5, seed_v = 1, debug = 0
                ):
    """ choose best parameters for data using grid search
    Args:
        X: input
        y: labels of input
        g_type_model (string): type of model
        g_input_shape (array): dimension of input
        g_num_classes (int): number of output of the network
        g_optimizer (string): optimizer function
        g_learning_rate (float): learning rate
        g_learning_decay (float): learning rate decay
        g_loss_func (string): loss function
        g_ml_number_filters (int): number of filters
        g_ml_numlayercnn_per_maxpool (int): number of cnn layer before each pooling
        g_ml_dropout_rate_fc (float): dropout at FC
        g_mc_nummaxpool (int): number of pooling layers
        g_mc_poolsize (int): pooling size
        g_mc_dropout_rate_cnn (float): dropout at CNN layers
        g_mc_filtersize (int): filter size  
        g_batch_size (int): mini batch size
        g_epochs_num (int): #epoch
        coef_test_ini (float): coefficient initialized at the beginning
        num_grid_coef (int): the number of coefficient to run grid search
        cv_time (int): cross validation for each coef
        seed_v (int): seed to shuffle data
        debug (int)

    Returns:
        the values of coefs used, the best coef; acc, auc of coefs
    """
    param_grid = []    
    acc_grid= []
    auc_grid=[]    
    
    max_temp=0
    index_max=0    
    max_auc_temp=0

    train_y1= y        
    train_y1=train_y1.reset_index()     
    train_y1=train_y1.drop('index', 1)

    for i in range(0,2):
    
        print('time: ' +str(i))
       
        param_grid.append((i+1))    
        
        acc_grid_cv,auc_grid_cv = cv_run_search(cv_time=cv_time,
            g_num_classes=(i+1),
            seed_v=seed_v, train_x1=X, train_y1=train_y1,debug=debug,
            g_type_model=g_type_model,g_input_shape=g_input_shape, 
            g_optimizer=g_optimizer,
            g_learning_rate=g_learning_rate,g_learning_decay=g_learning_decay, 
            g_loss_func=g_loss_func,
            g_ml_number_filters=g_ml_number_filters,
            g_ml_numlayercnn_per_maxpool=g_ml_numlayercnn_per_maxpool, 
            g_ml_dropout_rate_fc=g_ml_dropout_rate_fc, 
            g_mc_nummaxpool=g_mc_nummaxpool, 
            g_mc_poolsize=g_mc_poolsize, 
            g_mc_dropout_rate_cnn=g_mc_dropout_rate_cnn, 
            g_mc_filtersize=g_mc_filtersize,
            g_batch_size=g_batch_size, g_epochs_num=g_epochs_num,
            g_mc_padding = g_mc_padding
            )

        mean_acc_cv = np.around( np.mean(acc_grid_cv,axis=0), decimals=5)
        mean_auc_cv = np.around( np.mean(auc_grid_cv,axis=0), decimals=5)

        acc_grid.append (mean_acc_cv)  
        auc_grid.append (mean_auc_cv)                 

        if  mean_acc_cv  > max_temp or (mean_acc_cv == max_temp and mean_auc_cv > max_auc_temp) :
            max_temp =  mean_acc_cv
            index_max = i
            max_auc_temp = mean_auc_cv
    return param_grid, acc_grid, auc_grid, index_max    

def call_model (type_model,m_input_shape, m_num_classes, m_optimizer,m_learning_rate, m_learning_decay,m_loss_func,
            ml_number_filters,ml_numlayercnn_per_maxpool, ml_dropout_rate_fc, 
            mc_nummaxpool, mc_poolsize, mc_dropout_rate_cnn, mc_filtersize, mc_padding,
            svm_c, svm_kernel, rf_n_estimators,rf_max_depth, m_pretrained_file,knn_n_neighbors, rf_max_features):
    """ choose model (this function will call other funcs)
    Args:
        type_model (string): type of model
        m_input_shape (array): dimension of input
        m_num_classes (int): number of output of the network
        m_optimizer (string): optimizer function
        m_learning_rate (float): learning rate
        m_learning_decay (float): learning rate decay
        m_loss_func (string): loss function
        ml_number_filters (int): number of filters
        ml_numlayercnn_per_maxpool (int): number of cnn layer before each pooling
        ml_dropout_rate_fc (float): dropout at FC
        mc_nummaxpool (int): number of pooling layers
        mc_poolsize (int): pooling size
        mc_dropout_rate_cnn (float): dropout at CNN layers
        mc_filtersize (int): filter size    
        m_pretrained_file (string): pretrained file path
        knn_n_neighbors: #neighbors for KNN
        rf_max_features
       
        
    Returns:
        model with architecture depending on selection of users: CNNs, MLP, LTSM, FC
    """
    if type_model== "fc_model":
        model=fc_model(m_input_shape, m_num_classes, 
                optimizers_func=m_optimizer, lr_rate=m_learning_rate, 
                lr_decay=m_learning_decay, loss_func= m_loss_func, 
                dropout_fc=ml_dropout_rate_fc)
    elif type_model== "svm_model":
        model=svm_model(svm_C=svm_c,svm_kernel=svm_kernel, num_output=m_num_classes)
    elif type_model== "rf_model":
        #model=rf_model(rf_n_estimators=rf_n_estimators, rf_max_depth=rf_max_depth , rf_random_state = rf_random_state)
        model=rf_model(rf_n_estimators=rf_n_estimators, rf_max_depth=rf_max_depth, rf_max_features= rf_max_features )
    
    elif type_model == "dtc_model":
        #model=rf_model(rf_n_estimators=rf_n_estimators, rf_max_depth=rf_max_depth , rf_random_state = rf_random_state)
        model=dtc_model(rf_max_depth=rf_max_depth, rf_max_features= rf_max_features )

    elif type_model== "gbc_model":        
        model=gbc_model(rf_n_estimators=rf_n_estimators , rf_max_depth=rf_max_depth , rf_max_features= rf_max_features )
    elif type_model== "knn_model":
        model=knn_model(knn_n_neighbors=knn_n_neighbors)
    elif type_model== "fc_model_log":
        model=fc_model_log(m_input_shape, m_num_classes, 
                optimizers_func=m_optimizer, lr_rate=m_learning_rate, 
                lr_decay=m_learning_decay, loss_func= m_loss_func, 
                dropout_fc=ml_dropout_rate_fc)
    elif type_model== "model_mlp":
        model=model_mlp(m_input_shape,m_num_classes,
            optimizers_func=m_optimizer, lr_rate=m_learning_rate, lr_decay=m_learning_decay,
            loss_func= m_loss_func,
            num_neurons=ml_number_filters, 
            numlayer_layer=ml_numlayercnn_per_maxpool, 
            dropout_fc=ml_dropout_rate_fc)
    elif type_model== "model_cnn":
        model=model_cnn(m_input_shape,m_num_classes,
            optimizers_func=m_optimizer, lr_rate=m_learning_rate, lr_decay=m_learning_decay,
            loss_func= m_loss_func,
            numfilter=ml_number_filters, 
            filtersize=mc_filtersize,
            numlayercnn_per_maxpool=ml_numlayercnn_per_maxpool,
            nummaxpool=mc_nummaxpool,
            maxpoolsize=mc_poolsize,
            dropout_cnn=mc_dropout_rate_cnn, dropout_fc = ml_dropout_rate_fc, padded = mc_padding) 
    elif type_model== "model_cnn1d":
        print('opt=' + str(m_optimizer))
        model=model_cnn1d(m_input_shape,m_num_classes,
            optimizers_func=m_optimizer,
            lr_rate=m_learning_rate, lr_decay=m_learning_decay,
            loss_func= m_loss_func,
            numfilter=ml_number_filters, 
            filtersize=mc_filtersize,
            numlayercnn_per_maxpool=ml_numlayercnn_per_maxpool,
            nummaxpool=mc_nummaxpool,
            maxpoolsize=mc_poolsize,
            dropout_cnn=mc_dropout_rate_cnn, dropout_fc=ml_dropout_rate_fc, padded = mc_padding) 
    elif type_model== "model_lstm":
        # model=models_def.model_lstm()
        model=model_lstm(input_reshape=m_input_shape,num_classes=int(m_num_classes), 
            optimizers_func=m_optimizer, lr_rate=m_learning_rate, 
            lr_decay=m_learning_decay, loss_func= m_loss_func,#'mae', 
            num_neurons=ml_number_filters, numlayer_layer=ml_numlayercnn_per_maxpool)
    elif type_model== "model_lstm_baseline":
        # model=models_def.model_lstm()
        model=model_lstm_baseline(input_reshape=m_input_shape,num_classes=int(m_num_classes), 
            optimizers_func=m_optimizer,loss_func= m_loss_func)
    elif type_model== "model_con1d_baseline":
        # model=models_def.model_lstm()
        model=model_con1d_baseline(input_reshape=m_input_shape,num_classes=int(m_num_classes), 
            optimizers_func=m_optimizer,loss_func= m_loss_func)
            
    elif type_model== "model_cnn4_dropout":
        model=model_cnn4_dropout(input_reshape=m_input_shape,num_classes=int(m_num_classes),
            optimizers_func=m_optimizer)
    elif type_model== "model_vgglike":
        model=model_vgglike(input_reshape=m_input_shape,num_classes=int(m_num_classes),
            optimizers_func=m_optimizer, lr_rate=m_learning_rate, 
            dropout_fc = ml_dropout_rate_fc, dropout_cnn=mc_dropout_rate_cnn,
            lr_decay=m_learning_decay,loss_func= m_loss_func,padded = mc_padding)
    elif type_model in  ['resnet50', 'vgg16','pretrained']:
        model=model_pretrained(name_model =type_model,
            num_classes=int(m_num_classes),optimizers_func=m_optimizer, 
            loss_func=m_loss_func,input_reshape=m_input_shape, pretrained_file = m_pretrained_file)

    return model

def cv_run_search(cv_time,seed_v, train_x1, train_y1,debug,
        g_type_model,g_input_shape, g_num_classes, g_optimizer,
        g_learning_rate,g_learning_decay, 
        g_loss_func,
        g_ml_number_filters,
        g_ml_numlayercnn_per_maxpool, 
        g_ml_dropout_rate_fc, 
        g_mc_nummaxpool, 
        g_mc_poolsize, 
        g_mc_dropout_rate_cnn,  g_epochs_num,  g_batch_size,
        g_mc_filtersize, g_mc_padding):
    
    acc_grid_cv = []
    auc_grid_cv = []

    skf_grid=StratifiedKFold(n_splits=cv_time, random_state=seed_v, shuffle= True)
    for index_gr, (train_indices_gr,val_indices_gr) in enumerate(skf_grid.split(train_x1, train_y1.x)):
        print("Training on fold " + str(index_gr+1) + "/"+str(cv_time)+"...")
        
        train_x_grid = []
        train_y_grid = []
        val_x_grid = []
        val_y_grid = []           
        
        train_x_grid, val_x_grid = train_x1 [train_indices_gr], train_x1[val_indices_gr]
        train_y_grid, val_y_grid = train_y1.x [train_indices_gr], train_y1.x[val_indices_gr]

        if debug > 0:
            print(train_y_grid)
            print(val_y_grid)
        
        if g_num_classes == 2:
            train_y_grid=kr.utils.np_utils.to_categorical(train_y_grid)
            val_y_grid = kr.utils.np_utils.to_categorical(val_y_grid)
        
        # fc: g_num_classes
        # mlp: g_ml_number_filters, g_ml_numlayercnn_per_maxpool, g_ml_dropout_rate_fc
        # cnn: g_ml_number_filters, g_ml_numlayercnn_per_maxpool, g_ml_dropout_rate_fc, g_mc_dropout_rate_cnn
        # cnn1d: g_ml_number_filters, g_ml_numlayercnn_per_maxpool, g_ml_dropout_rate_fc
        # ltsm: g_ml_number_filters
        model1 = call_model (type_model=g_type_model, 
            m_input_shape = g_input_shape, m_num_classes= g_num_classes, m_optimizer = g_optimizer,
            m_learning_rate = g_learning_rate, m_learning_decay=g_learning_decay, m_loss_func=g_loss_func,
            ml_number_filters=g_ml_number_filters,ml_numlayercnn_per_maxpool=g_ml_numlayercnn_per_maxpool, ml_dropout_rate_fc=g_ml_dropout_rate_fc, 
            mc_nummaxpool=g_mc_nummaxpool, mc_poolsize= g_mc_poolsize, mc_dropout_rate_cnn=g_mc_dropout_rate_cnn, mc_filtersize=g_mc_filtersize, mc_padding=g_mc_padding)
        print('coef tune: dim=')
        print(train_x_grid.shape)
        print(val_x_grid.shape)
        history1 = model1.fit(train_x_grid, train_y_grid, 
                epochs = g_epochs_num, 
                batch_size=g_batch_size, 
                validation_data=(val_x_grid, val_y_grid),
                shuffle=False)  

        Y_pred = model1.predict(val_x_grid, verbose=2)
        val_auc_score=roc_auc_score(val_y_grid, Y_pred)

        acc_grid_cv .append(history1.history['val_acc'] [g_epochs_num-1])   
        auc_grid_cv .append(val_auc_score)     

    return acc_grid_cv,auc_grid_cv

class EarlyStopping_consecutively(keras.callbacks.Callback):
    """
    Based on Keras.Callback.callbacks.EarlyStopping but modified slight
    This function aims to reduce overfitting, stopping the training if val_loss is not improved 
        after pairs of consective previous-current epoch.
    
    This solves the problem with the training like this example: 
    (eg. patience=2, with Keras.Callback.callbacks.EarlyStopping the training will stop at epoch 4
        but with EarlyStopping_consecutively the training will continue until epoch 11)
        epoch 1: Val_loss 0.250
        epoch 2: Val_loss 0.100
        epoch 3: Val_loss 0.200
        epoch 4: Val_loss 0.192
        epoch 5: Val_loss 0.182
        epoch 6: Val_loss 0.150
        epoch 7: Val_loss 0.120
        epoch 8: Val_loss 0.080
        epoch 9: Val_loss 0.050
        epoch 10: Val_loss 0.055
        epoch 11: Val_loss 0.055

    """

    def __init__(self,
                monitor='val_loss',
                min_delta=0,
                patience=0,
                verbose=0,
                mode='auto'):
        super(EarlyStopping_consecutively, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        #modified
        self.previous=0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                            'fallback to auto mode.' % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

            if self.monitor_op == np.greater:
                self.min_delta *= 1
            else:
                self.min_delta *= -1

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' % (self.monitor),
                            RuntimeWarning)

        #modified, get the val_loss of the firt epoch
        if epoch==0:
            self.previous=logs.get(self.monitor)
    
        
        #if val_loss of current < previous, set wait=0
        #if self.monitor_op(current - self.min_delta, self.best):
        if self.monitor_op(current - self.min_delta, self.previous):
        # if current < self.previous:       
            self.wait = 0
        else: #if val_loss of current > previous, that means performance pause improving, then set wait+=1
            if epoch > 0: #skip the first epoch
                self.wait += 1 #add one to waited epochs
                print('----Current loss: ' +str(current) + ', previous loss: ' +str(self.previous) + '=> not improved! wait:' +  str(self.wait))
                if self.wait >= self.patience: #if wait reach limitation, then stopping training
                    self.stopped_epoch = epoch
                    self.model.stop_training = True      
        if epoch>1:
            self.previous=logs.get(self.monitor)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(('Epoch %05d: early stopping' % (self.stopped_epoch+1)))
        # print 'epoch early ' + str(epoch) + " cur" + str(current)
