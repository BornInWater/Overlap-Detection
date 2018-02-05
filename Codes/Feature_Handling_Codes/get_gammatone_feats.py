from scipy.io.wavfile import read as wavread
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import gammatonegram_package as gmtpack
import htkmfc as htk
import glob

#%matplotlib inline

# ----- use below to read filenames from list
#path = '/home/neeks/Desktop/Documents/dBase/TIMIT/'
#with open('/home/neeks/Desktop/Documents/dBase/TIMIT/all_files_wav.list') as f:
#    lines = f.read().splitlines()
#audiofilename = path+lines[0]

# ----- usebelow to read filenames from a directory
path = '/home/neeks/work/NEW_REGIME/SID/WAV/train/'
f = glob.glob("/home/neeks/work/NEW_REGIME/SID/WAV/train/*wav")
nfiles = len(f)
for loop in range(nfiles):
    
    filename = f[loop][len(path):-4]
    print(filename)
    [fs, x] = wavread(f[loop]) # x is a numpy array of integer, representing the samples 

    # scale to -1.0 -- 1.0
    if x.dtype == 'int16':
        nb_bits = 16 # -> 16-bit wav files
    elif x.dtype == 'int32':
        nb_bits = 32 # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    x = x / (max_nb_bit + 1.0) 

    # ----- gammatonegram specifications
    numChannels=64
    lowfreq=50
    TWIN = 0.025
    THOP = 0.010

    # ----- compute short-time gammatone transform (STGT)
    Y = gmtpack.get_gammatonegm_(x,fs,TWIN,THOP,lowfreq,numChannels)
    # ----- write in htk format
    #filename = "dummy_file"
    #writer=htk.open(filename+'.htk',mode='w',veclen=64) 
