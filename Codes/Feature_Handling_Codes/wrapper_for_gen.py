import re
import numpy as np
import generator as gen

f=open('/home/sriram/speech/OverLap/TIMIT/lists/train_files.list')
f=f.read()
f=f.strip()
f=re.split('\n',f)

for i in range(20000):
#for i in range(2):
        [index1,index2]=np.random.choice(3200,size=2,replace=False)
        file1=f[index1] 
#        file1='MBMA1_X414'
#        file2='MLEL0_X76'
        file2=f[index2]
        gen.gen_func(file1,file2,i)
        print "Completed Generating",  i+1 ,"Sentences"

