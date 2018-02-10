'''
This is a code to implement front end cnn processing followed by LSTM.
The model comprises of one LSTM layer + 3 Dense, whose input is fed from a CNN.
'''

import h5py
import htkmfc as htk
import keras
import keras.backend as K
import numpy as np
import scipy.io as sio
import sys
import theano
import time

from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.layers import Dense,Convolution2D,Dropout,MaxPooling2D,Input,Flatten,Activation,Merge,Dropout,LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional,TimeDistributed
from keras.models import Sequential,Model
from keras.models import load_model
from keras.optimizers import Adam,SGD
from keras.utils import np_utils

np.random.seed(2308)
perc=.5

#three command line arguments
train_data_path = sys.srgv[1] #path to training data
val_data_path = sys.argv[2] #path to validation data
context_size = sys.argv[3] #context size taken during data prep, eg. 11 for a left/right frame shift of 5

def Data_Getter(trainfile,valfile):
        print('Getting and prepping data')
        
        train=htk.open(trainfile)
        train_data=train.getall()
        np.random.shuffle(train_data)
        print('train_done')

        val=htk.open(valfile)
        val_data=val.getall()
        np.random.shuffle(val_data)
        print('val_done')

        Y_train=train_data[:,-1]
        X_train=train_data[:,:-1]
        del train_data
        time.sleep(5)
        Y_train=Y_train.reshape(Y_train.shape[0],1)
        Y_train=Y_train.astype(np.int8)
        Y_train=np_utils.to_categorical(Y_train,3)

	
        Y_val=val_data[:,-1]
        X_val=val_data[:,:-1]
        del val_data
        time.sleep(5)
        Y_val=Y_val.reshape(Y_val.shape[0],1)
        Y_val=Y_val.astype(np.int8)
        Y_val=np_utils.to_categorical(Y_val,3)

        print(X_train.shape,X_val.shape,Y_train.shape,Y_val.shape)
        return (X_train,X_val,Y_train,Y_val)


#to reduce learning rate and to load best model till now
class Master_Controller(keras.callbacks.Callback):
        def on_train_begin(self,logs={}):
                self.val_acc=[]
                self.best_val=0
        def on_epoch_end(self,epoch,logs={}):
                print('\n')
                print(' lr',model.optimizer.lr.get_value())
                model.optimizer.momentum.set_value(model.optimizer.momentum.get_value()+np.float32(.025))
                print('momentum value',model.optimizer.momentum.get_value())
                self.val_acc.append(logs.get('val_acc'))

		print('best_val:',self.best_val)

		if self.val_acc[epoch]<=self.best_val:
			file=h5py.File('best_model_cnn-rnn.h5','r')
			weight = []
			for i in range(len(file.keys())):
	   			weight.append(file['weight'+str(i)][:])
			model.set_weights(weight)
		
			print('the best model uptil now loaded again')
			model.optimizer.lr.set_value(K.cast_to_floatx(model.optimizer.lr.get_value()/2))	
		else:
			self.best_val=self.val_acc[epoch]
			file = h5py.File('best_model_cnn-rnn.h5','w')
			weight = model.get_weights()
			for i in range(len(weight)):
    				file.create_dataset('weight'+str(i),data=weight[i])
			file.close()
			print('validation accuracy increased and hence model saved')



#Finding mean and variance
def mean_var_finder(data):

        mean=np.mean(data,axis=0)
        var=np.var(data,axis=0)
        return (mean,var)

def cnn_reshaper(Data):

        Data=np.reshape(Data,(Data.shape[0],1,1,context_size,-1))
        return Data

def normalizer(data,mean,var):
        data=(data-mean)/np.sqrt(var)
        return data

csv_logger=CSVLogger('cnn-rnn.log')
lr_red=Master_Controller()

Context,Context_test,Y_train,Y_test=Data_Getter(train_data_path,val_data_path)
Context=cnn_reshaper(Context)
Context_test=cnn_reshaper(Context_test)
mean,var=mean_var_finder(Context)
Context=normalizer(Context,mean,var)

Context_test=normalizer(Context_test,mean,var)
sio.savemat('mean_var_cnn-rnn.mat',{'mean':mean,'var':var})

shape = Context.shape
filters = shape[1]/context_size

model=Sequential()

model.add(TimeDistributed(Conv2D(64,(5,7)),input_shape=(1,1,context_size,filters)))
model.add(TimeDistributed(Activation('relu')))
model.add(TimeDistributed(Flatten()))

model.add(LSTM(512))
model.add(Dense(1024,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.summary()

sgd=SGD(lr=.04,momentum=0.5,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
model.fit(Context,Y_train,nb_epoch=10,batch_size=256,validation_data=(Context_test,Y_test),callbacks=[lr_red])
scores=model.predict(Context_test,batch_size=512)

sio.savemat('cnn-rnn.mat',{'scores':scores,'Y_test':Y_test})
print('mat file name is cnn-rnn.mat')



