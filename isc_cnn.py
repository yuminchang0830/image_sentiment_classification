
m __future__ import print_function

import keras
from keras.models import Sequential
from keras.optimizers import RMSprop, SGD
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l1, l2
from keras.constraints import maxnorm
from keras import backend as K


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

batch_size = 100
epochs = 25
num_classes = 7


def plot_image(image):
	fig = plt.gcf()
	fig.set_size_inches(2,2)
	plt.imshow(image, cmap='binary')
	plt.show()

def load_data(train_data_path):
    train_data = pd.read_csv(train_data_path, sep=',', header=0)
    y_train = np.array(train_data['label'].values)
    x_train_object= np.array(train_data['feature'].values)
    x_train_array = []
    for i in range(0, x_train_object.size):
        results = str(x_train_object[i]).split(" ")
        results = [int(i) for i in results]        
        x_train_array.append(results)
    x_train = np.array(x_train_array)
    return (x_train, y_train)


def split_set(X, y, valid=0.1):
    Xt, yt, Xv, yv = [], [], [], []
    for idx, filename in enumerate(X):
        if random.random() < valid:
            Xv.append(X[idx])
            yv.append(y[idx])
        else:
            Xt.append(X[idx])
            yt.append(y[idx])

    assert len(X) == len(Xt) + len(Xv)        
    return Xt, yt, Xv, yv    


if K.image_data_format() == 'channels_first':
    input_shape = (1, 48, 48)
else:    
    input_shape = (48, 48, 1)

x_train, y_train = load_data("train.csv")


print ('x_train shape:',  x_train.shape)
print ('x_train dtype:', x_train.dtype)

x_train = x_train.astype('float32')
y_train = y_train.astype('float32')
x_train /= 255

x_train, y_train_label, x_test, y_test_label = split_set(x_train, y_train)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train_label = np.array(y_train_label)
y_test_label = np.array(y_test_label)

y_test = y_test_label

y_train_label = np_utils.to_categorical(y_train_label)
y_test_label = np_utils.to_categorical(y_test_label)

print ('y_train_label shape:',  y_train_label.shape)
print ('y_train_label dtype:', y_train_label.dtype)


print (y_test.shape)

'''
x_train_image = x_train.reshape(x_train.shape[0], 48, 48)
plot_image(x_train_image[3])
'''

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)

print ('x_train shape:', x_train.shape)
print ('x_train dtype:', x_train.dtype)
print ('x_test shape:', x_test.shape)
print ('x_test dtype:', x_test.dtype)
print ('y_train_label shape:',y_train_label.shape)
print ('y_train_label shape:',y_train_label.dtype)
print ('y_test_label shape:',y_test_label.shape)
print ('y_test_label shape:',y_test_label.dtype)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


'''
model = Sequential()
model.add(Dense(2304, activation='sigmoid', input_shape=(2304,)))
model.add(Dense(1152, activation='sigmoid'))
model.add(Dense(576, activation='sigmoid'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
'''              

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


#lrate = 0.01
#decay = lrate/epochs
#sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])             


history = model.fit(x_train, y_train_label,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)



traing_score = model.evaluate(x_train, y_train_label, verbose=0)
print('Traing loss:', traing_score[0])
print('Traing acc:', traing_score[1])

test_score = model.evaluate(x_test, y_test_label, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])

prediction=model.predict_classes(x_test)
print('------cross matrix----')
print('\n')
print(pd.crosstab(y_test, prediction, rownames=['label'], colnames=['predict']))


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

'''
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''


