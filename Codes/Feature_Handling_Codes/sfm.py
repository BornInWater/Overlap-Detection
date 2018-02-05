import re
import scipy.io.wavfile as wav
import scipy.io as sio
import numpy as np
import scipy.stats as stat
import wave
import math
import sklearn.preprocessing as skp
from scipy import signal as sig
rawlist='/home/neerajs/work/GMM/val.list'
wavfiledir='/home/data/amicorpus/'
savedir='/home/neerajs/work/GMM/whaterver_i_need/ami/feats/sfm/val/'


f=open(rawlist)
f=f.read()
f=f.strip()
f=re.split('\n',f)
eps=1e-9
for j in range(len(f)):
    print(j)
    raw=wave.open(wavfiledir+f[j]+'/audio/'+f[j]+'.Array1-01.wav','r')
    nchannels,sampwidth,sampling_rate,total_frames,comptype,compname=raw.getparams()
    sampling_rate,data=wav.read(wavfiledir+f[j]+'/audio/'+f[j]+'.Array1-01.wav','r')     
   # print "The size of the raw data: ",data.shape
    #print i," of ",len(f)
    if nchannels==1:
        signal=data
    else:
        signal=data[:,0]
    #some definitions
    duration=2.5e-2 #25ms is defined to be the duration for FFT
    shift_interval=1.0e-2 #duration
    samples=int(math.ceil(sampling_rate*duration)) #These are the number of array entries that we'll use to find the sfmosis
    skip_entries=int(math.ceil(sampling_rate*shift_interval)) #These entries are going to be skipped, that is we'll move to the next frame byb leaving these entries
    # columns=int(math.ceil(total_frames/skip_entries))
    # print signal.shape
    sfm_vals=[]
    # signal=skp.normalize(signal)
    iterator=0 #Just an iterator to control start and end points
    length=0 #Keeps track of the length that we have covered
    frames=0 #Keeps track of the number of frames
    while length<total_frames:
        start=iterator*skip_entries
        end=samples+start
        if end<total_frames:
            vector_for_sfm=signal[start:end]
            [freqs,psd]=sig.periodogram(vector_for_sfm,512)
            sfm_vals.append(stat.gmean(psd+eps)/(np.mean(psd)+eps))
            length=end
        else:
            vector_for_sfm=signal[start:total_frames]
            [freqs,psd]=sig.periodogram(vector_for_sfm,512)
            sfm_vals.append(stat.gmean(psd+eps)/(np.mean(psd)+eps))
            length=total_frames
        iterator+=1
        frames+=1
    #Done with the loop
    sfm_vals=np.asarray(sfm_vals)
    sfm_val=sfm_vals[0:frames-1]
    # wri=htk.open(savdirec+re.split("[*.*]",f[j])[0]+'.htk',mode='w',veclen=frames-1)
    # print sfm_vals.shape
    # wri.writeall(sfm_val)
    sio.savemat(savedir+f[j]+'.mat',{'sfm':sfm_val})
