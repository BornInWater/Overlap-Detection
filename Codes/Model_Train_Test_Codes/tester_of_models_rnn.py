'''
Testing code for rnn.py
'''
import h5py
import htkmfc as htk
import keras
import numpy as np
import scipy.io as sio
import sys
import time

from keras.models import Sequential,Model
from keras.layers import Dense,Convolution2D,Dropout,MaxPooling2D,Input,Flatten,Activation,Merge,Dropout,LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.models import load_model
from keras.layers.wrappers import Bidirectional,TimeDistributed

frames = sys.argv[2] #context size 

#loading data for testing
def data_getter(testfile):
	
	print 'getting and prepping data'
	val = htk.open(testfile)
	val_data = val.getall()
	Y_test = val_data[:,-1]
	X_test = val_data[:,:-1]
	del val_data
	time.sleep(5)
	Y_test = Y_test.reshape(Y_test.shape[0],1)
	Y_test = Y_test.astype(np.int8)

	return X_test,Y_test

#finding mean and var
def mean_var_finder(data):
	
	mean = np.mean(data,axis=0)
	var = np.var(data,axis=0)

	return (mean,var)


def rnn_reshaper(data):

	data=np.reshape(data,(data.shape[0],frames,64))
		       
	return data


def normalizer(data,mean,var):

	data = (data-mean)/np.sqrt(var)
	
	return data


def rnn_model(context,Y_test):
	
	model = Sequential()
	model.add(LSTM(512,input_shape=(frames,64)))
	model.add(Dense(1024,activation='relu'))
	model.add(Dense(512,activation='relu'))
	model.add(Dense(256,activation='relu'))
	model.add(Dense(3,activation='softmax'))

	model.compile(loss = 'categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
	
	file = h5py.File('best_model.h5')
	weight = []
	for i in range(len(file.keys())):
		weight.append(file['weight'+str(i)][:])
		model.set_weights(weight)

	scores = model.predict(context,batch_size=512)
	classes = model.predict_classes(context,batch_size=512)
	classes = np.reshape(classes,(classes.shape[0],1))

	Y_test = np.transpose(Y_test)

	sio.savemat('Predictions.mat',{'scores':scores,'classes':classes,'Y_test':Y_test})



if __name__ == '__main__':
	
	testfile = sys.argv[1]  #path of file to be tested
	context,Y_test = data_getter(testfile)

	data = sio.loadmat('mean_var.mat')
	mean = data['mean']
	var = data['var']

	context = rnn_reshaper(context)
	context = normalizer(context,mean,var)

	rnn_model(context,Y_test)



