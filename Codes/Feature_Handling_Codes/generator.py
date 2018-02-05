#The idea is to just generate the wav file and the overlap speech ground truth.
#We'll take 100ms as the offset, that would be ignored.
import re
import numpy as np
import scipy.io.wavfile as wav
import scipy
import scipy.io as sio

base='/home/neerajs/work/NEW_REGIME/WAV/MEETINGS/'
clean='clean_wav/'
rev='rev_wav/'
rev_noise='rev_noise_wav/'
rev_inaud='rev_inaud_wav/'

wavesavdir='/home/neerajs/work/NEW_REGIME/DATA/WAV/meeting/train/'
over_addr='/home/neerajs/work/NEW_REGIME/DATA/OVERLAP/meeting/train/'
#The place where the overlap files are going to be stored, which contain 0 if it is no overlap and 1 if there is overlap in the sample
choices=[clean,rev,rev_noise,rev_inaud]
#function to read in two wav files and two vad files and generate one wav file of overlapped speech and the corespoding vad file

# Right now, we'll assume that TIMIT data has no non speech regions. We'll just generate the Overlap files
# vad_addr='/home/neerajs/work/NEW_REGIME/VAD/original/'

phndir='/home/neerajs/work/NEW_REGIME/DATA/phones/train/'

def gen_func(file1,file2,i):
        print('Begin')
        print(file1,file2)
        #Fetching the base directory of the working files
        wav_addr1=base+choices[np.random.randint(3)]
        wav_addr2=base+choices[np.random.randint(4)]
        #reading the two wav files and overlapping them
        wav_file1=wav_addr1+file1+'.wav'
        # print(wav_file1)
        wav_file2=wav_addr2+file2+'.wav'
        # print(wav_file2)
        phnFile1=phndir+file1+'.PHN'
        phnFile2=phndir+file2+'.PHN'
	
	#### Check if the first line begins with 0 for h#
        nsFile1=[]
        phnFile1Fid=open(phnFile1).readlines()
	
	if int(phnFile1Fid[0].split(' ')[0]) != 0:
		nsFile1.append([0,int( phnFile1Fid[0].split(' ')[1])/160])

	for line in phnFile1Fid:
		line=line.rstrip()
		if re.search(('h#|epi|sil'),line):
       			nsFile1.append([int(line.split(' ')[0])/160,int(line.split(' ')[1])/160])

        # print nsFile1
 
        nsFile2=[]
        phnFile2Fid=open(phnFile2).readlines()
	
	if int(phnFile2Fid[0].split(' ')[0]) != 0:
		nsFile2.append([0,int( phnFile2Fid[0].split(' ')[0])/160])


	for line in phnFile2Fid:
		line=line.rstrip()
		if re.search(('h#|epi|sil'),line):
       			nsFile2.append([int(line.split(' ')[0])/160,int(line.split(' ')[1])/160])

        # print nsFile2
 
        #Wavfile.read returns the sampling rate and the read data. The sampling rate is assumed to be 16KHz for our purposes.
        [a1,a2]=wav.read(wav_file1)
        a2=np.reshape(a2,(1,a2.shape[0])) #a2 is the actual data sample, reshaping it to (1,size)
        [b1,b2]=wav.read(wav_file2)
        b2=np.reshape(b2,(1,b2.shape[0])) #b2 is the sample, and reshaping it

        #Making them floats
        a2=a2.astype(float)
        b2=b2.astype(float)

        # print "A2.shape: ",a2.shape
        # print "B2.shape: ",b2.shape
        #Chosing the time for overlap, 
        overlap_time=float(np.random.choice(np.arange(5,20),1))/10  # Pick a random overlap between 500ms and 2s
        print "Random time chosen for overlap: ",overlap_time
        overlap_sample=min(int(overlap_time*16000),min(a2.shape[1],b2.shape[1])) #The number of samples that are going to be overlapped
        overlap_frame=int(overlap_sample/160) # Number of frames of overlap (not taken silence into account)...
        
        overlap_part=a2[:,-overlap_sample:]+b2[:,0:overlap_sample]
        part1=a2[:,:-overlap_sample]
        part3=b2[:,overlap_sample:]
        out=np.hstack((part1,overlap_part,part3))#Actually creating the numpy array which has the overlap and single speaker speech segments
        #### FRAME LEVEL MANIPULATIONS FOR CREATING OVERLAP LABELS ####
        nFrames1=int(a2.shape[1]/160)
        nFrames2=int(b2.shape[1]/160) 
        # print nFrames1,nFrames2,overlap_frame,nsFile1,nsFile2

        stFrame2=nFrames1-overlap_frame
        totalFrame=nFrames1+nFrames2-overlap_frame-1
        # print stFrame2,totalFrame,nsFile2[0][0]+stFrame2 
        labelFrames1=np.ones((totalFrame,), dtype=np.int)
        for i in range(len(nsFile1)):
		labelFrames1[nsFile1[i][0]:nsFile1[i][1]] = 0
        labelFrames1[nFrames1:]=0
        labelFrames2=np.ones((totalFrame,), dtype=np.int)
        labelFrames2[:stFrame2]=0
        for i in range(len(nsFile2)):
                labelFrames2[nsFile2[i][0]+stFrame2:nsFile2[i][1]+stFrame2] = 0
        labelFrames = np.add(labelFrames1,labelFrames2)
	
 	out=out.astype(np.int16)
        out=np.reshape(out,(out.shape[1],1)) #Reshaping it to form the vector which is required to be written in wav file
        # print "The shape of out vector: ",out.shape
        scipy.io.wavfile.write(wavesavdir+file1+'-'+file2+'-'+str(overlap_sample)+'.wav',a1,out)
         
        scipy.io.savemat(over_addr+file1+'-'+file2+'-'+str(overlap_sample)+'.mat',{'labels':labelFrames})
# gen_func('MPGR0_I1410','MTPF0_I1865')
