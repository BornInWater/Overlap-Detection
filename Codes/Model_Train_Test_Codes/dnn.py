'''
Code to implement a neural network with three dense layers
followed by a softmax layer
'''
import keras
import numpy as np
import htkmfc as htk
import scipy.io as sio
import time
import keras
import keras.backend as K
import theano
import h5py
from keras.models import Sequential,Model
from keras.layers import Dense,Convolution2D,Dropout,MaxPooling2D,Input,Flatten,Activation,Merge,Dropout,LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional,TimeDistributed
from keras.utils import np_utils
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.models import load_model
from sklearn.model_selection import train_test_split
import sys
from theano.tensor import _shared

np.random.seed(2308)


#command line arguments
train_data_path = sys.srgv[1] #path to training data
val_data_path = sys.argv[2] #path to validation data

#to load train and validation data
def Data_Getter(trainfile,valfile):
	print('Getting and Prepping Data')
        train=htk.open(trainfile)
        train_data=train.getall()
        np.random.shuffle(train_data)
        print('train data loaded')
        print(train_data.shape)


        val=htk.open(valfile)
        val_data=val.getall()
        np.random.shuffle(val_data)
        print('val data loaded')
        print(val_data.shape)


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


# to reduce learning rate and to load best validation model till now
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
			file=h5py.File('best_model.h5','r')
			weight = []
			for i in range(len(file.keys())):
	   			weight.append(file['weight'+str(i)][:])
			model.set_weights(weight)
		
			print('the best model uptil now loaded again as no increase in validation accuracy')
			model.optimizer.lr.set_value(K.cast_to_floatx(model.optimizer.lr.get_value()/2))	
		else:
			self.best_val=self.val_acc[epoch]
			file = h5py.File('best_model.h5','w')
			weight = model.get_weights()
			for i in range(len(weight)):
    				file.create_dataset('weight'+str(i),data=weight[i])
			file.close()
			print('validation accuracy increased and hence model saved')





#finding mean and variance
def mean_var_finder(data):

        mean=np.mean(data,axis=0)
        var=np.var(data,axis=0)
        return (mean,var)

def normalizer(data,mean,var):
        data=(data-mean)/np.sqrt(var)
        return data

csv_logger=CSVLogger('log_file.log')
lr_red=Master_Controller()
Context,Context_test,Y_train,Y_test=Data_Getter(train_data_path,val_data_path)

mean,var=mean_var_finder(Context)
Context=normalizer(Context,mean,var)

Context_test=normalizer(Context_test,mean,var)
print('saving mean and var')
sio.savemat('mean_var_combined10_dense.mat',{'mean':mean,'var':var})

#model structure
model=Sequential()
model.add(Dense(256,input_shape=(704,),activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(3,activation='softmax'))
model.summary()


sgd=SGD(lr=.04,momentum=0.5,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
checkpointer=ModelCheckpoint(filepath='best_model.h5',monitor='val_acc',verbose=1,save_best_only=True,save_weights_only=True)
model.fit(Context,Y_train,nb_epoch=10,batch_size=256,validation_data=(Context_test,Y_test),callbacks=[lr_red])

scores=model.predict(Context_test,batch_size=512)
sio.savemat('Predictions.mat',{'scores':scores, 'Y_test': Y_test})


