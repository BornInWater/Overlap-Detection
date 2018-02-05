'''
This code computes the accuracy of nueral network predictions and displays the values in 
a confusion matrix
'''

import numpy as np
import os
import scipy.io as sio
import sys


def get_accuracy(classes,Y_test):
	
	print 'Validation accuracy ', float(sum(classes==Y_test))/len(Y_test)
	zero = np.where(Y_test==0)[0]
	one = np.where(Y_test==1)[0]
	two = np.where(Y_test==2)[0]

	single_ground = Y_test[one,:]
	overlap_ground = Y_test[two,:]

	ground = np.vstack((single_ground,overlap_ground))
	ground = ground.reshape(ground.shape[0],1)
	
	silence = classes[zero,:]
	single = classes[one,:]
	overlap = classes[two,:]

	label = np.vstack((single,overlap))
	label = label.reshape(label.shape[0],1)
	
	# 3x3 confusion matrix
	print('Validation Accuracy after removal of silence',float(sum(label==ground))/len(ground))
	print('silence accuracy',float(sum(silence==0))/len(zero),float(sum(silence==1))/len(zero),float(sum(silence==2))/len(zero))
	print('single accuracy',float(sum(single==0))/len(one),float(sum(single==1))/len(one),float(sum(single==2))/len(one))
	print('single accuracy',float(sum(overlap==0))/len(two),float(sum(overlap==1))/len(two),float(sum(overlap==2))/len(two))
	print(sum(label==ground),len(ground))


if __name__ == "__main__":
	File = sys.argv[1] 
	a = sio.loadmat(File)
	classes = a['classes'] #predicted
	Y_test = a['Y_test'] #ground truth
	Y_test = np.transpose(Y_test)
	get_accuracy(classes,Y_test)
