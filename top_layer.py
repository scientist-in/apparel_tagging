#learning
from sklearn import preprocessing
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import SGD

#python utils
import h5py
import numpy as np
import os
import re
import ipdb

ipdb.set_trace()
if os.path.abspath(__file__)=='/home/keeda/Documents/scientist/demo/apparel_tagging/top_layer.py':
    project_root = '/home/keeda/Documents/scientist/demo/apparel_tagging/'
else:
    project_root = '/home/ubuntu/apparel_tagging/'

training_batch_size = 10
nb_epoch = 10
#image_label_file
image_label_file =project_root+'fashion-data/train.txt'

#get labels
with open(image_label_file,'r') as f:
    image_labels = f.readlines()
image_labels = [x.strip('\n') for x in image_labels] 
p = re.compile('/(.*)/')
labels = [p.findall(i)[0] for i in image_labels]

#labels preprocessing
nb_classes = len(set(labels))
le = preprocessing.LabelEncoder()
le.fit(list(set(labels)))
labelEncoded = le.transform(labels)
labels_one_hot = np_utils.to_categorical(labelEncoded,nb_classes)

#order feature files in as per labels
features_folder = project_root+'features/'
feature_files = [features_folder+i for i in os.listdir(features_folder)]
p = re.compile('features(\d{1,2}?)-')
file_nos = [int(p.findall(filename)[0]) for filename in feature_files]
files_ordered = zip(feature_files,file_nos)
files_ordered.sort(key = lambda t:t[1])
feature_files = [i[1] for i in files_ordered]


#get features in loop and train
for idx, feature_file in enumerate(feature_files):
    with h5py.File(feature_files,'r') as hf:
        data = hf.get('features_h5')
        np_data_temp = np.array(data)
    print('Shape of the array features: \n', np_data_temp.shape)
    if not idx==0:
        np_data = np.append(np_data,np_data_temp,axis = 0)
    else:
        np_data = np_data_temp

    #define top layer
model = Sequential()
#input shape as per features
model.add(Flatten(input_shape = np_data.shape[1:4]))
model.add(Dense(nb_classes,activation='softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
if os.path.isfile('weights.h5'):
   model.load_weights('weights.h5') 

#training

model.fit(x=np_data,y=labels_one_hot,nb_epoch=nb_epoch,batch_size=training_batch_size)
#model.predict(np_data,batch_size=10)

#save_weights
model.save_weights('weights.h5')
    
    
    