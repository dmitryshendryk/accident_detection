from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import SGD, Adam
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications.vgg16 import VGG16
from keras.layers import LSTM, TimeDistributed, Bidirectional, GlobalMaxPool1D
from keras.models import load_model
import numpy as np
import glob,os
from scipy.misc import imread,imresize
from keras import backend as K

from sklearn.metrics import precision_score, accuracy_score

from keras.regularizers import l2

import argparse

batch_size = 128

def bring_data_from_directory():
  datagen = ImageDataGenerator(rescale=1./255)
  train_generator = datagen.flow_from_directory(
          '../train',
          target_size=(224, 224),
          batch_size=batch_size,
          class_mode='binary',  # this means our generator will only yield batches of data, no labels
          shuffle=True,
          classes=['Accidents','No_Accidents'])
  # print(train_generator.classes[3001])
  # print(train_generator.filenames[3001])
  validation_generator = datagen.flow_from_directory(
          '../validate',
          target_size=(224, 224),
          batch_size=batch_size,
          class_mode='binary',  # this means our generator will only yield batches of data, no labels
          shuffle=True,
          classes=['Accidents','No_Accidents'])

  return train_generator,validation_generator

def load_VGG16_model():
  base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
  print ("Model loaded..!")
  print (base_model.summary())
  return base_model

def extract_features_and_store(train_generator,validation_generator,base_model):
  # x_generator = None
  # y_lable = None
  # batch = 0
  # for x,y in train_generator:
  #     if batch == int(560/batch_size):
  #         break
  #     print("Total needed:", int(56021/batch_size))
  #     print ("predict on batch:",batch)
  #     batch+=1
  #     if np.any(x_generator)==None:
  #        x_generator = base_model.predict_on_batch(x)
  #        y_lable = y
  #        print (y)
  #     else:
  #        x_generator = np.append(x_generator,base_model.predict_on_batch(x),axis=0)
  #        y_lable = np.append(y_lable,y,axis=0)
  #        print (y)
  # x_generator,y_lable = shuffle(x_generator,y_lable)
  # np.save(open('video_x_VGG16.npy', 'wb'), x_generator)
  # np.save(open('video_y_VGG16.npy','wb'),y_lable)
  # batch = 0
  # x_generator = None
  # y_lable = None
  # for x,y in validation_generator:
  #     if batch == int(397/batch_size):
  #         break
  #     print("Total needed:", int(3971/batch_size))
  #     print ("predict on batch validate:",batch)
  #     batch+=1
  #     if np.any(x_generator)==None:
  #        x_generator = base_model.predict_on_batch(x)
  #        y_lable = y
  #        print (y)
  #     else:
  #        x_generator = np.append(x_generator,base_model.predict_on_batch(x),axis=0)
  #        y_lable = np.append(y_lable,y,axis=0)
  #        print (y)
  # x_generator,y_lable = shuffle(x_generator,y_lable)
  # np.save(open('video_x_validate_VGG16.npy', 'wb'),x_generator)
  # np.save(open('video_y_validate_VGG16.npy','wb'),y_lable)

  train_data = np.load(open('video_x_VGG16.npy', 'rb'))
  train_labels = np.load(open('video_y_VGG16.npy', 'rb'))
  train_data,train_labels = shuffle(train_data,train_labels)
  print(train_data)
  validation_data = np.load(open('video_x_validate_VGG16.npy', 'rb'))
  validation_labels = np.load(open('video_y_validate_VGG16.npy', 'rb'))
  validation_data,validation_labels = shuffle(validation_data,validation_labels)

  train_data = train_data.reshape(train_data.shape[0],
                     train_data.shape[1] * train_data.shape[2],
                     train_data.shape[3])
  validation_data = validation_data.reshape(validation_data.shape[0],
                     validation_data.shape[1] * validation_data.shape[2],
                     validation_data.shape[3])
  
  return train_data,train_labels,validation_data,validation_labels

def train_model(train_data,train_labels,validation_data,validation_labels):
  ''' used fully connected layers, SGD optimizer and 
      checkpoint to store the best weights'''
  print("SHAPE OF DATA : {}".format(train_data.shape))
  model = Sequential()
  model.add(LSTM(4096, kernel_initializer='glorot_uniform', bias_initializer='zeros',   dropout=0.2, input_shape=(train_data.shape[1],
                     train_data.shape[2])))
  model.add(Dense(1024, kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1, kernel_initializer='random_uniform', bias_initializer='zeros', activation='sigmoid'))
  adam = Adam(lr=0.00005, decay = 1e-6)
  model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
  #model.load_weights('video_1_LSTM_1_512.h5')
  callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=0), ModelCheckpoint('video_1_LSTM_1_1024.h5', monitor='val_loss', save_best_only=True, verbose=0) ]
  nb_epoch = 500
  model.fit(train_data,train_labels,validation_data=(validation_data,validation_labels),batch_size=batch_size,nb_epoch=nb_epoch,callbacks=callbacks,shuffle=True,verbose=1)
  return model


def detection(lstm_weights, test_dataset):

  x = []
  y = []
  output = 0
  print("Load base BGG16 model")
  base_model = load_VGG16_model()
  print("Load LSTM model")
  model = load_model(lstm_weights)
  print(model.summary())

  images = os.listdir(test_dataset)

  for img_name in images:
    image = imread(test_dataset + '/' + img_name)
    image = imresize(image , (224,224))

    x.append(image)
    y.append(output)
  
  
  x = np.array(x)
  y = np.array(y)
  x_features = base_model.predict(x)
  x_features = x_features.reshape(x_features.shape[0], x_features.shape[1]*x_features.shape[2], x_features.shape[3])
  answer = model.predict(x_features)
  correct = 0
  
  answer = [int(np.round(x)) for x in answer]


  for i in range(len(answer)):
    if y[i] == answer[i]:
        correct+=1
  print(correct,"correct",len(answer))
  print("Accuracy is {}%".format((correct/len(answer)) *100))





if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument('--weights')
  parser.add_argument('--dataset')

  parser.add_argument('command', metavar='<command>')

  args = parser.parse_args()

  if args.command == 'train':
    train_generator,validation_generator = bring_data_from_directory()
    base_model = load_VGG16_model()
    train_data,train_labels,validation_data,validation_labels = extract_features_and_store(train_generator,validation_generator,base_model)
    train_model(train_data,train_labels,validation_data,validation_labels)
  # test_on_whole_videos(train_data,train_labels,validation_data,validation_labels)

  if args.command == 'detect':
    detection(args.weights, args.dataset)



