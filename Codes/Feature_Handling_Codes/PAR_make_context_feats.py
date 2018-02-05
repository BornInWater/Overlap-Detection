
# Code to:
# a. make context features by concatenating context_size frames
# b. parallel implementation
# -----
# Last modified: 10th July, 2017, Neeks
# Created by: Neeks
# -----

import numpy as np
import htkmfc as htk
import glob
import re
import scipy.io as sio
import sys
from joblib import Parallel, delayed
import multiprocessing


# inits
database='timit'
context_size = 5
feature_type = 'six_fb'
data_nature = 'val'
# update the path to respective directories
label_val_addr = '/home/neerajs/work/NEW_REGIME/DATA/OVERLAP/val/'
fbank_feats_addr = '/home/neerajs/work/NEW_REGIME/DATA/FEATS/fbank_after/val/'
context_addr = '/home/neerajs/work/NEW_REGIME/DATA/FEATS/context/'

# valing file list
a=open('/home/neerajs/work/NEW_REGIME/DATA/FEATS/context/fbank/temp.list')
data=a.read()
data=data.strip()
filename=re.split('\n',data)

#with open(list_path) as f:
#    useful_files = f.read().splitlines()

#filename = []
#for i in range(len(useful_files)):
#    temp = useful_files[i].replace('.', '_')
#    filename.append(temp)





def context_gen(i):
    print('File #: ' +str(i+1)+ '/' +str(len(filename)))
    
    # load the label MAT file
    
    labels_addr = label_val_addr + filename[i] + '.mat'
    a = sio.loadmat(labels_addr)
    vad = a['labels']

    if database == 'timit':
	vad=np.transpose(vad)

    # load the datafile
    if((feature_type.find('fbank'))!=-1):
        data_read = htk.open(fbank_feats_addr+filename[i]+'.htk')
        data = data_read.getall()        
        data = np.transpose(data)
        op_path = 'fbank'
    elif((feature_type.find('six_fb'))!=-1):
        data_read = htk.open(fbank_feats_addr+filename[i]+'.htk')
        data = data_read.getall()        
        data = np.transpose(data)
        op_path = 'mfcc';
    elif((feature_type.find('gamma'))!=-1):
        data_read = htk.open(fbank_feats_addr+filename[i]+'.htk')
        data = data_read.getall()        
        data = np.transpose(data)
        op_path = 'gamma';
    else:
        print('Feature type incorrect! Execution terminates.')
        sys.exit()

    print('Filename: '+filename[i])
    print('Feature Dimension: ' + str(data.shape[0])+ ' X ' + str(data.shape[1]))
  
    # scan the frames and make context features
    nframes = min(data.shape[1],vad.shape[0])
    data_write = np.zeros((nframes,(2*context_size+1)*data.shape[0]),np.float)
    label_write = np.zeros((nframes,1),np.float)

    ssegs = 0
    for index in range(nframes):
        # make left/right context
        if (index<context_size):
            temp_l = np.zeros((data.shape[0],context_size),np.float)
            temp_r = data[:,index+1:index+1+context_size]
        elif ((nframes-index)<(context_size+1)):
            temp_l = data[:,index-context_size:index]
            temp_r = np.zeros((data.shape[0],context_size),np.float)
        else:
            temp_r = data[:,index+1:index+1+context_size]
            temp_l = data[:,index-context_size:index]

        # fill left/right context by mirror flipping only if empty
        if (temp_l.sum()==0):
            temp_l = np.fliplr(temp_r)
        elif (temp_r.sum()==0):
            temp_r = np.fliplr(temp_l)

        # complete the context frame with the center frame
        mid = data[:,index]
        mid = mid.reshape(mid.shape[0],1)

        data_temp = np.hstack((temp_l, mid, temp_r))
        data_write[ssegs,:] = data_temp.reshape(1,(2*context_size+1)*data.shape[0])

        # label

        label_write[ssegs,:] = vad[index,:];
        ssegs = ssegs + 1; 

    # save the context feats of each file
    if database== 'timit':

	    writer=htk.open(context_addr+ op_path+ '/val/'+ filename[i]+'.htk',mode='w',veclen=(2*context_size+1)*data.shape[0]) 
	    writer.writeall(data_write)    
	    writer.close()
	    writer=htk.open(context_addr + 'labels/'+'val/' +filename[i]+'.htk',mode='w',veclen=1) 
	    writer.writeall(label_write)    
	    writer.close()  
    else:
            writer=htk.open(context_addr + 'val/' + op_path +'/' +filename[i] + '.htk', mode='w',veclen=(2*context_size+1)*data.shape[0])
            writer.writeall(data_write)
            writer.close()
            writer=htk.open(context_addr + 'val/' + '/labels/' +filename[i]+'.htk',mode='w',veclen=1)
            writer.writeall(label_write)
            writer.close()
			    
num_cores=multiprocessing.cpu_count()
print(num_cores)
Parallel(n_jobs=num_cores/8)(delayed(context_gen)(i) for i in range(len(filename)))

      
    
