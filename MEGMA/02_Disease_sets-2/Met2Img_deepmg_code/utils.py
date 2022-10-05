"""
======================================================================================
UTILS functions
======================================================================================
Author: Thanh Hai Nguyen
date: 20/12/2017 (updated to 31/10/2018, stable version)'
'this module includes:
'1. find_files: find files based on a given pattern, aiming to avoid repeating the experiments
'2. load_img_util: load images
'3. textcolor_display: show messages in the screen with color 
"""

#from scipy.misc import imread
import numpy as np
import os, fnmatch
import math

def find_files(pattern, path):
    """ find files in path based on pattern
    Args:
        pattern (string): pattern of file
        path (string): path to look for the files
    Return 
        list of names found
    """
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

def load_img_util (num_sample,path_write,dim_img,preprocess_img,channel,mode_pre_img, pattern_img='train'):
    """ load and reading images save to data array 
    Args:
        data_dir (array): folder contains images
        pattern_img (string): the pattern names of images (eg. 'img_1.png, img_2.png,...' so pattern='img') 
        num_sample (int) : the number of images read
        dim_img: dimension of images
        preprocess_img: preprocessing images, support: vgg16, vgg19, resnet50, incep
        path_write: path to save images

    Returns:
        array
    """
    from keras.preprocessing import image
    from keras.applications.resnet50 import preprocess_input as pre_resnet50
    from keras.applications.vgg16 import preprocess_input as pre_vgg16
    from keras.applications.vgg19 import preprocess_input as pre_vgg19
    from keras.applications.inception_v3 import  preprocess_input as pre_incep
    
    temp = []
    for i in range(0,num_sample): #load samples for learning
        image_path = os.path.join(path_write, str(pattern_img) +str("_") + str(i) +".png")               
        if dim_img==-1: #if use real img
            if channel == 4:
                #img = imread(image_path)
                #img = img.astype('float32')
                print ('waiting for fixing issues from from scipy.misc import imread')
                exit()
            elif channel == 1:
                img = image.load_img(image_path,grayscale=True)
            else:   
                img = image.load_img(image_path)
        else: #if select dimension
           
            if channel == 1:
                img = image.load_img(image_path,grayscale=True, target_size=(dim_img, dim_img))
            
            else:
                img = image.load_img(image_path,target_size=(dim_img, dim_img))

        x = image.img_to_array(img)         
        # x = preprocess_input(x)
        if preprocess_img=='vgg16':
            x = pre_vgg16(x, mode= mode_pre_img)
        elif preprocess_img=='resnet50':
            x = pre_resnet50(x, mode= mode_pre_img)
        elif preprocess_img=='vgg19':
            x = pre_vgg19(x, mode= mode_pre_img)
        elif preprocess_img=='incep':
            x = pre_incep(x)
    
        temp.append(x)      

    return np.stack(temp)             

def textcolor_display(text,type_mes='er'):
    '''
    show text in format with color
    Agr:
        text: text to format with color
        type_mes : type of messgage could be 'er' or 'inf'
    return 
        text formatted with color
    '''
    end = '\x1b[0m'
    if type_mes in ['er','error']:
        begin = '\x1b[1;33;41m'        
        return begin + text + end
    
    if type_mes in ['inf','information']:
        begin = '\x1b[0;36;44m'        
        return begin + text + end