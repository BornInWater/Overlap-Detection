#file to concat features from multiple files into one big matrix for creating train/val/test data


import re
import os
import sys
import time
import numpy as np
import scipy.io as sio
import htkmfc as htk
	
def file_opener(file_read):
        file_reader=open(file_read)
        file_reader=file_reader.read()
        file_reader=file_reader.strip()
        file_reader=re.split('\n',file_reader)
        return file_reader

def data_creator(num,addr,file_reader,filename):
    	corrupt_files=0
	ind=0
    	writer=htk.open(filename+'.htk',mode='w',veclen=num)
	for i in range(int(len(file_reader))):
        	print(i)
            	data_read=htk.open(addr+file_reader[i]+'.htk')
            	try:
                	read_data=data_read.getall()

		except:
			corrupt_files+=1
			continue

		ind=ind+read_data.shape[0]
		print(read_data.shape)
		writer.writeall(read_data)
	print('corrput_files',corrupt_files)

if os.path.isdir('Data'):
	pass
else:
	os.mkdir('Data')
os.chdir('Data')


#addr='/home/neerajs/work/feats/' #address where features are stored
addr='/home/neerajs/work/labels/' #address where labels are stored
num=1  # length of feature/label [eg: 40 for mfcc, 1 for labels]
file_read='/home/neerajs/work/lists/list.list' # file which contains filenames whose features are to be concatenated 
filename='concat' #filename for storing concatenated matrix

file_reader=file_opener(file_read)
data_creator(num,addr,file_reader,filename)

