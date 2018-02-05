
import numpy as np
import gammatonegram_package as gmtpack
from scipy import signal

def ERBSpace_(lowFreq=50,highFreq=8000,N=64):
    EarQ = 9.26449
    minBW = 24.7
    order = 1

    temp = (-np.log(highFreq + EarQ*minBW) +  np.log(lowFreq + EarQ*minBW))/N

    cfArray = -(EarQ*minBW) + np.exp(temp*np.arange(1,N+1))*(highFreq+EarQ*minBW)

    return cfArray

def MakeERBFilters_(fs=16000,numChannels=64,lowfreq=50):
    T = 1/float(fs)
    cf = ERBSpace_(lowfreq,fs/2,numChannels)

    EarQ = 9.26449
    minBW = 24.7
    order = 1.0
    ERB = ((cf/EarQ)**order+minBW**order)**(1/order)
    B = 1.019*2*np.pi*ERB

    A0 = T
    A2 = 0.0
    B0 = 1.0
    B1 = -2*np.divide(np.cos(2*cf*np.pi*T),np.exp(B*T))
    B2 = np.exp(-2*B*T)

    A11 = -(np.divide(2*T*np.cos(2*cf*np.pi*T),np.exp(B*T)) + np.divide(2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T),np.exp(B*T)))/2
    A12 = -(np.divide(2*T*np.cos(2*cf*np.pi*T),np.exp(B*T)) - np.divide(2*np.sqrt(3+2**1.5)*T*np.sin(2*cf*np.pi*T),np.exp(B*T)))/2
    A13 = -(np.divide(2*T*np.cos(2*cf*np.pi*T),np.exp(B*T)) + np.divide(2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T),np.exp(B*T)))/2
    A14 = -(np.divide(2*T*np.cos(2*cf*np.pi*T),np.exp(B*T)) - np.divide(2*np.sqrt(3-2**1.5)*T*np.sin(2*cf*np.pi*T),np.exp(B*T)))/2


    temp1 = -2*np.exp(4*1j*cf*np.pi*T)*T + np.multiply(2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T,\
                       np.cos(2*cf*np.pi*T) - np.sqrt(3 - 2**(3/2))*np.sin(2*cf*np.pi*T))
    temp2 = -2*np.exp(4*1j*cf*np.pi*T)*T + np.multiply(2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T,\
                       np.cos(2*cf*np.pi*T) + np.sqrt(3 - 2**(3/2))*np.sin(2*cf*np.pi*T))
    temp3 = -2*np.exp(4*1j*cf*np.pi*T)*T + np.multiply(2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T,\
                       np.cos(2*cf*np.pi*T) - np.sqrt(3 + 2**(3/2))*np.sin(2*cf*np.pi*T))
    temp4 = -2*np.exp(4*1j*cf*np.pi*T)*T + np.multiply(2*np.exp(-(B*T) + 2*1j*cf*np.pi*T)*T,\
                       np.cos(2*cf*np.pi*T) + np.sqrt(3 + 2**(3/2))*np.sin(2*cf*np.pi*T))
    temp5 = (np.divide(-2,np.exp(2*B*T)) - 2*np.exp(4*1j*cf*np.pi*T) + 2*np.divide(1+np.exp(4*1j*cf*np.pi*T),np.exp(B*T)))**4

    gain = np.multiply(temp1,np.multiply(temp2,np.multiply(temp3,temp4)))
    gain = np.abs(np.divide(gain,temp5))
    allfilts = np.ones((cf.shape[0],1),np.float)

    fcoefs = np.zeros((cf.shape[0],10),np.float)
    fcoefs[:,0] = A0*allfilts[:,0]
    fcoefs[:,1] = A11
    fcoefs[:,2] = A12
    fcoefs[:,3] = A13
    fcoefs[:,4] = A14
    fcoefs[:,5] = A2*allfilts[:,0]
    fcoefs[:,6] = B0*allfilts[:,0]
    fcoefs[:,7] = B1
    fcoefs[:,8] = B2
    fcoefs[:,9] = gain

    return fcoefs

def ERBFilterBank_(x,fcoefs):
    A0  = fcoefs[:,0]
    A11 = fcoefs[:,1]
    A12 = fcoefs[:,2]
    A13 = fcoefs[:,3]
    A14 = fcoefs[:,4]
    A2  = fcoefs[:,5]
    B0  = fcoefs[:,6]
    B1  = fcoefs[:,7]
    B2  = fcoefs[:,8]
    gain = fcoefs[:,9]

    output = np.zeros((gain.shape[0],x.shape[0]),np.float)
    for chan in range(gain.shape[0]):
        y1 = signal.lfilter([A0[chan]/gain[chan], A11[chan]/gain[chan], A2[chan]/gain[chan]],\
                                [B0[chan], B1[chan], B2[chan]],x)

        y2 = signal.lfilter([A0[chan], A12[chan], A2[chan]],\
                                [B0[chan], B1[chan], B2[chan]],y1)

        y3 = signal.lfilter([A0[chan], A13[chan], A2[chan]],\
                                [B0[chan], B1[chan], B2[chan]],y2)

        y4 = signal.lfilter([A0[chan], A14[chan], A2[chan]],\
                                [B0[chan], B1[chan], B2[chan]],y3)

        output[chan,:] = y4
        
    return output

def get_gammatonegm_(x,fs=16000,TWIN=0.025,THOP=0.01,lowfreq=50,numChannels=64):
    # ----- compute the filters
    fcoefs = MakeERBFilters_(fs,numChannels,lowfreq)
    fcoefs = np.flipud(fcoefs)

    # ----- filter the data
    XF = ERBFilterBank_(x,fcoefs)
    XF = XF**2.0

    # ----- short-time gammatone transform (STGT)
    nwin = int(TWIN*fs)
    hopsamps = int(THOP*fs)
    ncols = 1 + int((XF.shape[1]-nwin)/hopsamps)
    Y = np.zeros((numChannels,ncols),np.float)
    #indx = np.arange(0,nwin)

    for i in range(ncols):
        Y[:,i] = np.sqrt(np.mean(XF[:,(i-1)*hopsamps + np.arange(0,nwin)],axis=1))
    return Y