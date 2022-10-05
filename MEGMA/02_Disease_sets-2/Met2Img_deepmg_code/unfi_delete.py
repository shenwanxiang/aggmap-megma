'''to delete unfinished files:
Author: Nguyen Thanh Hai
Date: 23/03/2018

## Function: This module aims to delete "-m delete" or count (default) "-m count" unfinished/redundancy files. 

## Note: please backup before running this file. ONLY use if ensuring that there is NO any running job.

## Usage: copy these commands
cd ~/deepMG_tf/
python ./utils/read_results/unfi_delete.py

'''
import numpy as np
import os
from optparse import OptionParser

#read para from cmd
parser = OptionParser()
parser.add_option("-i", "--folder_parent_results", type="string", default='./', help="locate the parent folder where contains results") 
parser.add_option("-l", "--file_listname", type="string", default='delete_files', help="naming the file containing list of files") 
parser.add_option("-o", "--folder_file_listname", type="string", default='./', help="locate folder containing the file of list") 
parser.add_option("-m", "--mode_run", type='choice', choices=['count','delete'], default='count', help="mode of running: count: only counting, delete: deleting files") 
parser.add_option("-p", "--name_pattern", type='string', default='*sum.txt', help="pattern files for searching") 

(options, args) = parser.parse_args()

#save file name that UNfinished the experient (does not contain "ok" in the file1 name)
cmd1 = 'find '+ str(options.folder_parent_results)+' -name "'+str(options.name_pattern)+'" > ' + str(options.folder_file_listname) + str(options.file_listname) + '.txt'
print (cmd1)
os.system(cmd1)

#read file containing file name
file_general = open ( str(options.file_listname) +'.txt',"r" )  
list_filenames = file_general.readlines()

num_files = len(list_filenames)
count_deleted = 0
for i in range(0,num_files):    
    #get file name i
    file_name_deleted = list_filenames [i]
    print ('file ' + str(i+1) + ' ' + str(file_name_deleted))
    cmd1 = 'rm ' + str(file_name_deleted)
    print (cmd1)
    #there are 2 supported modes: deleted and count
    if options.mode_run=='deleted':
        os.system(cmd1)
    if options.name_pattern == '*file1.txt':
        cmd1 = 'rm ' + file_name_deleted[0:file_name_deleted.index('_sum.txt')] + str('_eachfold.txt')
        print (cmd1)
        if options.mode_run=='deleted':
            os.system(cmd1)

        cmd1 = 'rm ' + file_name_deleted[0:file_name_deleted.index('1.txt')] + str('3.txt')
        print (cmd1)
        if options.mode_run=='deleted':
            os.system(cmd1)
    count_deleted = count_deleted + 1

if options.mode_run=='deleted':
    print (str(count_deleted) + ' experiments, ' + str(count_deleted*3) + ' files deleted')
else:
    print (str(count_deleted) + ' experiments, ' + str(count_deleted*3) + ' files counted')
