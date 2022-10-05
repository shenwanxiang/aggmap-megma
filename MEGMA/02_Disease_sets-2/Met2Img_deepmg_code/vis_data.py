"""
======================================================================================
functions for visualizing data
======================================================================================
Author: Thanh Hai Nguyen, Team Integromics, ICAN, Paris, France'
date: 20/12/2017 (updated to 05/05/2019, stable version)'
'this module includes:
'1. convert_color_real_img: convert data to given hex value (for SPecies Bins (SPB) and PResent (PR) bins)
'2. convert_bin: convert data to index of bin
'3. embed_image: generating images using manifold learning (t-SNE, LDA, PCA...)
'4. fillup_image: generating images using Fill-up
'5. coordinates_fillup: build coordinates for fill-up
'6. coordinates_z_fillup: build coordinates for fill-up with Z shape
"""
#from scipy.misc import imread
try:
    from utils_deepmg import textcolor_display as text_display
except ImportError:
    from deepmg.utils_deepmg import textcolor_display as text_display

#from sklearn.cluster import KMeans
import numpy as np
import math
from sys import exit


import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
              
def build_breaks (data,type_bin,num_bin=10,max_v=1,min_v=0):
    '''
    building breaks to generate bins
    ARGS
        data (array-nD)
        type_bin (string): support 'eqf', 'eqw', 'sbp', 'pr' (add kmeans from Feb. 2020)
        max_v (float): combine with min_v to generate bins in eqw (used in 'eqw')
        min_v (float): values will be excluded when calculating bins (used in 'eqf','eqw','pr')
        num_bin (int): number of bins (used in 'eqf','eqw')
    RETURN 
        bins_break (array-1D) 
    '''
    bins_break = np.empty((0), float)      
    if type_bin == 'eqf':
        # step 1: flaten
        data_f = data.flatten()
        # step 2 remove 0 or values which are not counted as an appearance of feature in sample --> data_non0
        data_f_excluded = [i for i in data_f if i != min_v]
        # step 3: sort data_non0 ascending
        data_f_excluded_sorted = np.sort(data_f_excluded)
        # step 4: count of values excluding min_v     
        count_value = len(data_f_excluded_sorted)    

        # step 5: identify breaks based on #bin         
        # add first bin with first break which start to bin     
        bins_break = np.append(bins_break,min_v)      
        for i in range (0,num_bin-1):
            #print("int(count_value/10)*i=")
            #print(int(count_value/10)*i)
            bins_break = np.append(bins_break,(data_f_excluded_sorted [int(count_value/10)*i]))
    elif type_bin == 'eqw':
        dis_max_min =  (max_v - min_v)/ num_bin
        for i in range (0,num_bin):
            bins_break = np.append(bins_break, min_v + dis_max_min * i)
    elif type_bin == 'spb':
        bins_break = [
                      min_v,
                      0.0000001,
                      0.0000004,
                      0.0000016,
                      0.0000064,
                      0.0000256, 
                      0.0001024, 
                      0.0004096,
                      0.0016384, 
                      0.0065536]
    elif type_bin == 'pr':
        bins_break = [min_v]   

    ######## Author: nhi #########    
    elif type_bin == 'kmeans':
        # step 1: flaten
        data_f = data.flatten().reshape(-1, 1)
        # step 2 remove 0 or values which are not counted as an appearance of feature in sample --> data_non0
        data_f_excluded = [i for i in data_f if i != min_v]
        from sklearn.cluster import KMeans
        # step 3: Use K-means algorithm
        kmeans = KMeans(n_clusters = num_bin - 1).fit(data_f_excluded)
        # step 4: Calculator bins_break
        bins_break = get_bin_break(bins_break, data_f_excluded, kmeans.labels_)
    ######## end of Author: nhi ######### 
    elif type_bin == 'kmedoids':
        ##from pyclustering.cluster import kmedoids
        from KMedoids import KMedoids
        # step 1: flaten
        data_f = data.flatten().reshape(-1, 1)
        # step 2 remove 0 or values which are not counted as an appearance of feature in sample --> data_non0
        data_f_excluded = [i for i in data_f if i != min_v]
        # step 3: Use K-means algorithm
        KMedoids = KMedoids(n_cluster = num_bin - 1).fit(data_f_excluded)
        # step 4: Calculator bins_break
        bins_break = get_bin_break(bins_break, data_f_excluded, labels = KMedoids.labels_)


    else:
        print ( text_display( '--type_bin=' + str(type_bin) + ' is not available now!!'))
        exit()
    
    return bins_break


######## Author: nhi #########
def get_bin_break(bins_break, data_f_excluded, labels):
    '''
    :param bins_break: array bins break
    :param data: array data is flatten and remove 0 or values which are not counted as an appearance of feature in s
    :param labels: array labels of kmeans
    :return:
    '''

    result = np.empty((0), float)
    bins_break = np.append(bins_break, 0)
    data = []
    data_index = []

    for i in range(0, len(data_f_excluded)):
        data.append([i, data_f_excluded[i][0], labels[i]])

    data = sorted(data, key = lambda x : (x[2], x[1]))

    data_index.append(0)
    for i in range(1, len(data)):
        if data[i][2] != data[i- 1][2]:
            data_index.append(i)

    data = np.array(data)
    for i in range(0, len(data_index)):
        if i + 2 == len(data_index):
            result = np.append(result,
                               (max(data[data_index[i]:data_index[i + 1], 1]) + min(data[data_index[i + 1]:, 1])) / 2)

        elif i + 2 < len(data_index):
            result = np.append(result, (
                    max(data[data_index[i]:data_index[i + 1], 1]) + min(data[data_index[i + 1]:data_index[i + 2], 1])) / 2)
        else:
            break

    result = sorted(result)
    for i in result:
        bins_break = np.append(bins_break, i)

    bins_break = np.append(bins_break, 1)
    return bins_break

######## end of Author: nhi #########

#def convert_bin (value, num_bin=10,  min_v= 0,  color_img=False, bins_break = [] , debug=0):
def convert_bin (value, num_bin=10,min_v= 0,  color_img=False, bins_break = [] , debug=0):
    """ load and reading images save to data array  (use for abundance data) to create images
    **notes: ONLY apply to data range 0-1
    Args:
        value (float): orginial value
        debug (int): mode of debug        
        num_bin : number of bins
        min_v: value will be excluded from binning
        color_img : return hex if num_bin=10 and use color images 
    Return:
        a float if color_img=False, 
        or a hex if color_img=True
    """
    color_v = 0   
    #color_arry_num = [1,0.9,0.8,0.7,0.6, 0.5,0.4,0.3, 0.2,0.1]
    color_arry = ['#000000', '#8B0000','#FF0000','#EE4000', '#FFA500', '#FFFF00','#00CD00', '#0000FF','#00BFFF','#ADD8E6','#FFFFFF'] 
    num_bin = len(bins_break)
    #print(bins_break)
    if value > min_v:

        bin_index = [int(i+1) for i in range(0,num_bin) if ( math.isclose (bins_break[i],value) or bins_break[i]< value)]
        if debug==1:
            print (bin_index)
            print (num_bin)
        if bin_index == []:
            v_bin = float(1/num_bin)
        else:            
            v_bin = float((max(bin_index))/num_bin)    
    else:
        v_bin = 0     
    
    if color_img and num_bin==10: #if use 10 distinct color, return #hex
        color_v = color_arry [ int(num_bin - v_bin*num_bin)]
    else: #if use gray, return real value
        color_v =  v_bin

    if debug==1:
        print ('min_v=' +str(min_v))
        print (value,color_v)

    return color_v    
  
def bins_image (X,colormap,min_v,num_bin, bins_breaks = []):
    '''
    AGRS
        X (array): a sample with n features
        colormap (string): colormap used to generate the image
        min_v (float): the value will ignore when coloring
        num_bin (int): the number of bins used (#colors)
        color_img (bool): color or gray ?
        bins_breaks (array): array of breaks for binning
    return 
        array of binned value of the features in a sample 
    '''    
    if colormap == '':
        print( text_display('Please specify --colormap'))
        exit()
    elif colormap in ['custom']:
        return [convert_bin(value=y, min_v = min_v, num_bin = num_bin, bins_break = bins_breaks, color_img =True) for y in X ]
    else: #another colormap
        return [convert_bin(value=y, min_v = min_v, num_bin = num_bin, bins_break = bins_breaks) for y in X ]

def embed_image (X_embedded, X, color_arr,colormap, file_name_emb, 
        size_p, fig_size,marker_p,alpha_v,cmap_vmin, cmap_vmax,
        dpi_v = 75,margin = 0,off_axis= True,show= False):
    
    """ create an image using manifolds #https://matplotlib.org/api/markers_api.html
    Args:
        X_embedded (array): coordinates of points after manifold
        X (array) : value of data point 
        color_arr (array): array of colors of points
        file_name_emb (string): name image output
        size_p (int): point size
        fig_size=4 (int) : figure size (usually, 1: gives imgages of 84x84, 2: 162x162,...)
        type_bin: type of data: spb/pr/eqw
        marker (string): shape of point (refer:https://matplotlib.org/api/markers_api.html), should use 'o' (for large img and if density not high) and ',': pixels (for small images)
        alpha_v (float): mode of transparent 0-1
        num_bin (int): number of bins
        margin (float): margin of image (white border)
        
        min_v: mini value
        dpi_v : dpi of images
        colormap : colormap used to set colors for the image (if ''-->custom set)
        [cmap_vmin, cmap_vmax]: the range to set colors using colormap provided by Python
    Returns:
        an image
    """

       
    #set options to remove padding/white border
    mpl.rcParams['savefig.pad_inches'] = 0   
    fig, ax = plt.subplots(figsize=(fig_size*1.0/dpi_v,fig_size*1.0/dpi_v), dpi=dpi_v, facecolor='w')   
    #ax.set_axis_bgcolor('w')
    #ax.set_axis_bgcolor('w')
    ax.set_facecolor('w')


    
    if off_axis:  #eliminate border/padding for generating images
        ax.axis('off') #if do not have this, images will appear " strange black point"!!
        ax = plt.axes([0,0,1,1], frameon=False) #refer https://gist.github.com/kylemcdonald/bedcc053db0e7843ef95c531957cb90f
        # Then we disable our xaxis and yaxis completely. If we just say plt.axis('off'),
        # they are still used in the computation of the image padding.
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Even though our axes (plot region) are set to cover the whole image with [0,0,1,1],
        # by default they leave padding between the plotted data and the frame. We use tigher=True
        # to make sure the data gets scaled to the full extents of the axes.
        plt.autoscale(tight=True) 
        #print fig.get_dpi() #default
    else:
        ax.axis('on') 
        #plt.autoscale(tight=True) 
        

   
    
    #set lim_max,min in x-axis and y-axis
    x_max = np.max(X_embedded[:,0])
    x_min = np.min(X_embedded[:,0])
    y_max = np.max(X_embedded[:,1])
    y_min = np.min(X_embedded[:,1])
    print('x_max'+ str(x_max))
    print('x_min'+ str(x_min))
    print('y_max'+ str(y_max))
    print('y_min'+ str(y_min))
    #variable containing data point != 0 or not avaiable
    new_color_array = []
    new_X_embedded = []
    
    #if importances_feature != 'none':
    #    print 'use important feature'
    #     for i in range(0,len(importances_feature)):
    #         print X_embedded[importances_feature [i] ]
    #         print color_arr [importances_feature [i] ]

    #skip white points (which have no information)
    if colormap == "gray":   
        color_1 =  np.ones(len(color_arr)) - color_arr
        color_1 = [str(i) for i in color_1]
        #print color_1
        for i in range(0,int(len(X_embedded))) :
            if color_1[i] != '1.0': #in grayscale, '1': white
                new_color_array.append(color_1[i]) #convert to string and scale of gray
                new_X_embedded.append(X_embedded[i])    
    elif colormap =='custom':
        for i in range(0,int(len(X_embedded))) :
            if color_arr[i] != 'w':
                    new_color_array.append(color_arr [i])
                    new_X_embedded.append(X_embedded[i])   
    else:
        color_1 =  np.ones(len(color_arr)) - color_arr           
        for i in range(0,int(len(X_embedded))) :
            #print("i color=========",i,'len',len(X_embedded))
            if color_1[i] != 1.0: #in grayscale, 1: white
                    new_color_array.append(color_1[i]) #convert to string and scale of gray
                    new_X_embedded.append(X_embedded[i]) 
            

    new_X_embedded= np.stack(new_X_embedded)
    #print 'len(new_X_embedded)=' + str(len(new_X_embedded))
    #print 'len(new_color_array)=' + str(len(new_color_array))
    #print new_color_array
    #print len(new_X_embedded)
    if colormap in ['gray','custom']: #if use predefined color or grayscale
        ax.scatter(new_X_embedded[:,0],new_X_embedded[:,1], s=size_p, marker = marker_p,color=new_color_array, edgecolors='none', alpha = alpha_v)          
    else:       
        if not(colormap in ['viridis','rainbow','gist_rainbow','jet','nipy_spectral','Paired','Reds','YlGnBu',
                        'viridis_r','rainbow_r','gist_rainbow_r','jet_r','nipy_spectral_r','Paired_r','Reds_r','YlGnBu_r']):             
            print (  text_display( 'colormap ' +str(colormap) + ' is not supported!!') )
            exit()            
        #ax.scatter(new_X_embedded[:,0],new_X_embedded[:,1], s=size_p, marker = marker,c=new_color_array, edgecolors='none', alpha = alpha_v, cmap=cmap)        
        #ax.scatter(new_X_embedded[:,0],new_X_embedded[:,1], s=size_p, marker = marker,c=new_color_array, edgecolors='none', alpha = alpha_v, cmap=plt.get_cmap(colormap))        
        if cmap_vmax == cmap_vmin:
            ax.scatter(new_X_embedded[:,0],new_X_embedded[:,1], s=size_p, marker = marker_p,c=new_color_array, edgecolors='none', alpha = alpha_v, cmap=plt.get_cmap(colormap),vmin=cmap_vmax/num_bin,vmax=cmap_vmax)      
        else:
            ax.scatter(new_X_embedded[:,0],new_X_embedded[:,1], s=size_p, marker = marker_p,c=new_color_array, edgecolors='none', alpha = alpha_v, cmap=plt.get_cmap(colormap),vmin=cmap_vmin,vmax=cmap_vmax)      
  
    #fixing the same positions for all images  
    plt.xlim([x_min - margin, x_max + margin])
    plt.ylim([y_min - margin, y_max + margin])


    #create image
    if show :
        plt.show() 
    else:
        fig.savefig(file_name_emb+'.png')#, transparent=False) ##True to see "not available area (unused area)"  
        print (file_name_emb)
        

    plt.close('all')

  
def fillup_image (cor_x, cor_y, X, colors,colormap,fig_size,file_name, 
        size_p,marker_p,alpha_v,cmap_vmin, cmap_vmax,
        dpi_v = 75,margin = 0,show=False, off_axis = True):
    """ create an image using fillup 
    Args:       
        X (array): value of data point    
        file_name (string): name image output
        fig_size=4 (int) : figure size (usually, 1: gives imgages of 84x84, 2: 162x162,...), 
            to compute/convert inches-pixel look at http://auctionrepair.com/pixels.html, 
                Pixels / DPI = Inches
                Inches * DPI = Pixels
        size_p (int): point size
        type_bin (string): type of data (abundance: spb or presence: pr)        
        cor_x (float): coordinates of x
        cor_y (float): coordinates of y
        marker_p (string): shape of point (refer to :https://matplotlib.org/api/markers_api.html), should use 'o' (for large img and if density not high) and ',': pixels (for small images)
        min_v: mini value
        dpi_v (int): dpi of images
        colormap : colormap used to set colors for the image (if ''-->custom set)
            colormap for color images (refer to https://matplotlib.org/examples/color/colormaps_reference.html)
        [cmap_vmin, cmap_vmax]: the range to set colors using colormap provided by Python
        show (int): 0: not shown, !=0: show the image not generating (using in jupyter)
        
    Returns:
        an image

    Notes:
        options used in the function            
            
            alpha_v
            point_size
            shape_drawn            
            colormap
            cmap_vmin
            cmap_vmax
            off_axis (true): turn off the axis

    """
   
    
    #set the size of images
    mpl.rcParams['savefig.pad_inches'] = 0   
    fig, ax = plt.subplots(figsize=(fig_size*1.0/dpi_v,fig_size*1.0/dpi_v), dpi=dpi_v, facecolor='w')   
    #ax.set_axis_bgcolor('w')
    ax.set_facecolor('w')

    #eliminate border/padding
    if off_axis :
        ax.axis('off') #if do not have this, images will appear " strange black point"!!
        ax = plt.axes([0,0,1,1], frameon=False) #refer https://gist.github.com/kylemcdonald/bedcc053db0e7843ef95c531957cb90f
        # Then we disable our xaxis and yaxis completely. If we just say plt.axis('off'),
        # they are still used in the computation of the image padding.
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # Even though our axes (plot region) are set to cover the whole image with [0,0,1,1],
        # by default they leave padding between the plotted data and the frame. We use tigher=True
        # to make sure the data gets scaled to the full extents of the axes.
        plt.autoscale(tight=True) 
        #print fig.get_dpi() #default
    else:
        ax.axis('on') 
        plt.autoscale(tight=True) 

    
    
    #size_fig: eg. t2d: 23.9165214862 -> matrix 24x24
    #draw images
    if colormap == "gray":                     
        #ax.scatter(cor_x,cor_y, marker = marker_p, color=str( 1 - colors))
        color_1 =  np.ones(len(colors)) - colors
        color_1 = [str(i) for i in color_1]
        #print colors
        #print np.max(colors)
        #print color_1
        ax.scatter(cor_x,cor_y, s=size_p, marker = marker_p,color=color_1, edgecolors='none', alpha = alpha_v)

        #ax.plot(cor_x,cor_y, marker_p, color=str( 1 - colors), markersize=size_p)
    elif colormap in ['custom']: #if use predefined color       
        ax.scatter(cor_x,cor_y, s=size_p, marker = marker_p,color=colors, edgecolors='none', alpha = alpha_v)
    else: 
        #print (colors)
        # refer to https://pythonspot.com/matplotlib-scatterplot/
        if not(colormap in ['viridis','rainbow','gist_rainbow','jet','nipy_spectral','Paired','Reds','YlGnBu', 
                        'viridis_r','rainbow_r','gist_rainbow_r','jet_r','nipy_spectral_r','Paired_r','Reds_r','YlGnBu_r']):             
            print ( text_display ( 'colormap ' +str(colormap) + ' is not supported!!') )
            exit()
            
        #print 'colormap ' +str(colormap) + ' selected!'
        #colors = np.stack(colors)
        #print colors
        #color_1 =  np.ones(len(colors)) - colors
        
        #set lim_max,min in x-axis and y-axis
        x_max = np.max(cor_x)
        x_min = np.min(cor_x)
        y_max = np.max(cor_y)
        y_min = np.min(cor_y)

        cor_x_new =[]
        cor_y_new =[]
        color_new = []
        #skip value = 0
        for i in range(0,int(len(colors))) : #remove point with value = 0
            if colors[i] != 0.0: #in grayscale, 1: white
                color_new.append(1-colors[i]) #convert to string and scale of gray
                cor_x_new.append(cor_x[i]) 
                cor_y_new.append(cor_y[i]) 
        #color_1 = [str(i) for i in color_1]
        # ax.scatter(cor_x_new,cor_y_new, s=size_p, marker = marker_p,c=color_new, edgecolors='none', alpha = alpha_v,cmap=cmap)
        #ax.scatter(cor_x_new,cor_y_new, s=size_p, marker = marker_p,c=color_new, edgecolors='none', alpha = alpha_v,cmap=plt.get_cmap(colormap))
        #ax.scatter(cor_x_new,cor_y_new, s=size_p, marker = marker_p,c=color_new, edgecolors='none', alpha = alpha_v,cmap=plt.get_cmap(colormap),vmin=1.0/num_bin,vmax=1)
        if cmap_vmax == cmap_vmin:
            print ('please set cmap_vmin<>cmap_vmax')
            exit()
            #ax.scatter(cor_x_new,cor_y_new, s=size_p, marker = marker_p,c=color_new, edgecolors='none', alpha = alpha_v,cmap=plt.get_cmap(colormap),vmin=cmap_vmax/num_bin,vmax=cmap_vmax)
        else:              
            ax.scatter(cor_x_new,cor_y_new, s=size_p, marker = marker_p,c=color_new, edgecolors='none', alpha = alpha_v,cmap=plt.get_cmap(colormap),vmin=cmap_vmin,vmax=cmap_vmax)
        
        
        #this code to keep the same positions for all images belongging to a dataset
        #plt.xlim([x_min, x_max])
        #plt.ylim([y_min, y_max])
        plt.xlim([x_min - margin, x_max + margin])
        plt.ylim([y_min - margin, y_max + margin])
    
    #create image
    if show :
        plt.show() 

    else:
        fig.savefig(file_name+'.png')#, transparent=False) ##True to see "not available area (unused area)"  
        print (file_name)
        

    plt.close('all')

def coordinates_fillup (num_features):
    '''
    generating coordinates for fill-up based on the number of features
    '''
    cordi_x = []
    cordi_y = []   
    #build coordinates for fill-up with a square of len_square*len_square
    len_square = int(math.ceil(math.sqrt(num_features)))
    print ('square_fit_features=' + str(len_square) )
    k = 0
    for i in range(0,len_square):
        for j in range(0,len_square):                
            if k == (num_features):
                break
            else:
                cordi_x.append(j*(-1))
                cordi_y.append(i*(-1))
                k = k+1
        if k == (num_features):
            break
    print ('#features=' +str(k))
    return cordi_x, cordi_y
   
def coordinates_z_fillup (num_features):
    cordi_x = []
    cordi_y = []   
    #build coordinates for fill-up with a square of len_square*len_square
    len_square = int(math.ceil(math.sqrt(num_features)))
    print ('square_fit_features=' + str(len_square) )
    k = 0
    odd = 1
    for i in range(0,len_square):
        if odd % 2 == 1:
            for j in range(0,len_square):                
                if k == (num_features):
                    break
                else:
                    cordi_x.append(j*(-1))
                    cordi_y.append(i*(-1))
                    k = k+1
            if k == (num_features):
                break
        else:
            for j in range(len_square,0,-1):                
                if k == (num_features):
                    break
                else:
                    cordi_x.append(j*(-1))
                    cordi_y.append(i*(-1))
                    k = k+1
            if k == (num_features):
                break
        odd = odd + 1

    print ('#features=' +str(k))
    return cordi_x, cordi_y

def generate_image (data, bins_breaks, colormap, file_name_emb,size_p, fig_size,
    marker,alpha_v,cmap_vmin, cmap_vmax, 
    min_v, num_bin,
    X_embedded=[],cor_x=[],cor_y=[], v_show = False, v_off_axis = True, dpi_v = 75, margin = 0):
    '''
    description: generate bins, then create images
    arg:
        data: a sample
        bins_breaks: a 1D-array of breaks

    return:
        an image
    '''

    if ( (X_embedded == [] and cor_x == []) or (X_embedded != [] and cor_x != []) ):
        print ( text_display ('the coordinates of features are not valid, please check again!!') )
        exit()

    if (bins_breaks==[]):
        print (text_display('bins_breaks is empty. Please check again'))
        exit()

    colors_array = bins_image (X = data,colormap = colormap,min_v=min_v,num_bin=num_bin, bins_breaks = bins_breaks)
    
    
    
    if (X_embedded == []):
        #print ('using FILL-UP')
        fillup_image (cor_x=cor_x, cor_y=cor_y, X = data, colors = colors_array,
            colormap = colormap, fig_size = fig_size,file_name = file_name_emb, 
            size_p = size_p,marker_p = marker,alpha_v = alpha_v,cmap_vmin = cmap_vmin, cmap_vmax = cmap_vmax,margin = margin, show=v_show, off_axis = v_off_axis, dpi_v = dpi_v)
    elif (cor_x == [] or cor_y == []):
        #print ('using visualizations algorithms (except Fill-up')
        embed_image (X_embedded = X_embedded, X= data, color_arr = colors_array,
            colormap = colormap,fig_size = fig_size,file_name_emb = file_name_emb, 
            size_p = size_p, marker_p = marker,alpha_v = alpha_v,cmap_vmin = cmap_vmin, cmap_vmax = cmap_vmax,margin = margin, show=v_show, off_axis = v_off_axis, dpi_v = dpi_v)
    else:
        print ( text_display ('Generating images cannot be done. Please check again!!') )
        exit()
