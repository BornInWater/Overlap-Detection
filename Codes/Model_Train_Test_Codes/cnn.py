'''
Code to implement a CNN consisting of 3 convolutional layers
followed by a pooling which is fed to dense layers.
'''

import keras
import sys
import os
import numpy as np
import scipy.io as sio
import time
import htkmfc as htk
import keras.backend as K
from keras.models import Sequential,Model
from keras.layers import Dense,Conv2D,Dropout,MaxPooling2D,Input,Flatten,Activation,Merge
from keras.utils import np_utils
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras.models import load_model
from keras.callbacks import CSVLogger

np.random.seed(2308)

#command line arguments
train_data_path = sys.srgv[1] #path to training data
val_data_path = sys.argv[2] #path to validation data
context_size = sys.argv[3] #context size taken during data prep, eg. 11 for a left/right frame shift of 5

# to load train and validation data
def Data_Getter(trainfile,valfile):
	print('Getting and Prepping Data')
        train=htk.open(trainfile)
        train_data=train.getall()
	np.random.shuffle(train_data)
        print('train data loaded')

        val=htk.open(valfile)
        val_data=val.getall()
	np.random.shuffle(val_data)
        print('validation data loaded')

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
	print 'Shapes of train and val data'
	print(X_train.shape,X_val.shape,Y_train.shape,Y_val.shape)
        return (X_train,X_val,Y_train,Y_val)

#to reduce learning rate and to load best validation model till now
class Master_Controller(keras.callbacks.Callback):
        def on_train_begin(self,logs={}):
                self.val_acc=[]
                self.count=0
                self.start=0
                self.end=0
                self.best_val=0
        def on_epoch_end(self,epoch,logs={}):
                print('\n')
                print(' lr',model.optimizer.lr.get_value())
                model.optimizer.momentum.set_value(model.optimizer.momentum.get_value()+np.float32(.025))
                print('momentum value',model.optimizer.momentum.get_value())
                self.val_acc.append(logs.get('val_acc'))

                if epoch>1:
                        print('best_val:',self.best_val)

                        if self.val_acc[epoch]<=self.best_val:
                                model.load_weights('best_model.h5')
                                print('no increase in validation accuracy; best model loaded')
                                if self.count==0:
                                        self.start=epoch
                                        self.end=epoch
                                        self.count=1
                                else:
                                        self.end=epoch
                                        if(self.end-self.start<10):
                                                model.optimizer.lr.set_value(K.cast_to_floatx(model.optimizer.lr.get_value()/2))
                                                self.count=0
                        else:
                                self.best_val=self.val_acc[epoch]




#finding mean and variance 
def mean_var_finder(data):

	mean=np.mean(data,axis=0)
	var=np.var(data,axis=0)
	
	return (mean,var)

def cnn_reshaper(Data):

	Data=np.reshape(Data,(Data.shape[0],1,context_size,-1))
	return Data

def normalizer(data,mean,var):
	data=(data-mean)/np.sqrt(var)
	return data

csv_logger = CSVLogger('log_file.log')
lr_red = Master_Controller()

Context,Context_val,Y_train,Y_val=Data_Getter(train_data_path,val_data_path)
Context=cnn_reshaper(Context)
Context_val=cnn_reshaper(Context_val)
mean,var=mean_var_finder(Context)
Context=normalizer(Context,mean,var)
Context_val=normalizer(Context_val,mean,var)

print('saving mean and var')
sio.savemat('mean_var.mat',{'mean':mean,'var':var})

#defining the sequential model
model=Sequential()
model.add(Conv2D(64,(5,7),input_shape=(1,context_size,64)))
model.add(Activation('relu'))
model.add(Conv2D(128,(3,5)))
model.add(Activation('relu'))
model.add(Conv2D(256,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.summary()


sgd=SGD(lr=.04,momentum=0.5,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
checkpointer=ModelCheckpoint(filepath='best_model.h5',monitor='val_acc',verbose=1,save_best_only=True,save_weights_only=True)
model.fit(Context,Y_train,epochs=10,batch_size=1024,validation_data=(Context_val,Y_val),callbacks=[lr_red,checkpointer,csv_logger])

scores=model.predict(Context_test,batch_size=512)
sio.savemat('Predictions.mat',{'scores':scores,'Y_test':Y_test})

