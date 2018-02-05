

# Code to:
# a. do CMVN
# b. operates on the files in a specified folder
# c. stores the features using HTK format
# ------
# Last updated: 10th July, 2017 (Neeks)
# Created by: Neeks
# ------

import glob
import htkmfc as htk
import multiprocessing
import numpy as np
import sys
from joblib import Parallel, delayed

# features storage folder
# Provide the path of the raw features as well as the storage path for features after cmvn
feature_addr = sys.argv[1]
store_addr = sys.argv[2]

# input the feature folder
# modify extension as required
f = glob.glob(feature_addr+'*htk')

nfiles = len(f)
print 'Number of files:', nfiles
def context_gen(i):

    # get feature filename
    filename = f[i][len(feature_addr):-4]
    print(filename)
    
    # read using htk: nframes X channels
    data_read=htk.open(feature_addr+filename+'.htk')
    x=data_read.getall()
    
    # apply cmvn
    varnorm = 1
    # get mean across time along each channel
    mu = np.mean(x,0)
    mu = mu.reshape(1,mu.shape[0])

    # get standard deviation across time along each channel
    eps = np.spacing(np.float32(1.0))
    if (varnorm == 1):
        stddev = np.std(x,0)
        stddev = stddev.reshape(1,stddev.shape[0])
    else:
        stddev = 1
    y = (x-mu)/(stddev+eps) # uses broadcasting for element-wise division
    
    # store feature
    writer=htk.open(store_addr+filename+'.htk',mode='w',veclen=y.shape[1]) 
    writer.writeall(y)    
    writer.close()
    

num_cores=multiprocessing.cpu_count()
Parallel(n_jobs=num_cores/8)(delayed(context_gen)(i) for i in range(nfiles))
