#learning
from vgg16 import VGG16
from inception_v3 import InceptionV3
from resnet50 import ResNet50
from vgg19  import VGG19
from keras.models import Model

#learning utils
from keras.preprocessing import image
from imagenet_utils import preprocess_input

#python utils
import ipdb
import numpy as np
import sys
from sys import getsizeof
import cPickle as pickle
import h5py
import re

if os.path.abspath(__file__)=='/home/keeda/Documents/scientist/demo/apparel_tagging/chk_aws.py':
    project_root = '/home/keeda/Documents/scientist/demo/apparel_tagging/'
else:
    project_root = '/home/ubuntu/apparel_tagging/'

#prep image locations
image_path_file = project_root+'fashion-data/train.txt'
images_loc = project_root+'fashion-data/images/'
with open(image_path_file) as f:
    image_paths = f.readlines()
image_paths = [images_loc+x.strip('\n')+'.jpg' for x in image_paths] 

#select indices for breaking the data into chunks (image_data variable can otherwise get huge)
img_chunk = 350 #images
elements_in_temp = 10
batch_size_for_pred = 4

features_indices = list()
for i in range(len(image_paths)/img_chunk+1):
    features_indices = features_indices + [(i*img_chunk,i*img_chunk+len(image_paths)%img_chunk-1 if (i+1)*img_chunk>(len(image_paths)) else (i+1)*img_chunk-1)]
with h5py.File('feature_indices.h5', 'w') as hf:
    hf.create_dataset('feature_indices_h5', data=features_indices)



#select model

#model = VGG16(weights='imagenet', include_top=False)
#model = VGG19(weights= 'imagenet',include_top = False)
#model = InceptionV3(weights='imagenet', include_top=False)
model = ResNet50(weights= 'imagenet', include_top =False)

#create feature files
for ind,i,j in enumerate(features_indices):
    

    #prep first image
    img = image.load_img(image_paths[i], target_size=(224, 224))
    img = image.img_to_array(img)
    image_data_temp = np.expand_dims(img, axis=0)
    image_data = image_data_temp
    
    
    #prep rest of the image
    element_num = j+1
    elements_in_temp = elements_in_temp
    batch_size_for_pred = batch_size_for_pred
    for idx,image_path in enumerate(image_paths[i+1:element_num]):
        if (idx+1)%elements_in_temp==0:
            image_data = np.append(image_data,image_data_temp,axis=0)
            print idx
            #print image_data.shape
            if element_num<len(image_paths):
                img = image.load_img(image_paths[idx+1], target_size=(224, 224))
                img = image.img_to_array(img)
                image_data_temp = np.expand_dims(img, axis=0)
            continue
        img = image.load_img(image_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        if idx == 0:
            image_data_temp = img
        else:
            image_data_temp = np.append(image_data_temp,img,axis=0)
        print idx
        
    #ipdb.set_trace()  
    if image_data.shape[0]<element_num:
        image_data = np.append(image_data,image_data_temp,axis = 0)
    #ipdb.set_trace()
    
    features = model.predict(image_data,batch_size = batch_size_for_pred, verbose =1)
    
    #ipdb.set_trace()
    with h5py.File(project_root+'features/'+'features'+ ind+'-'+str(i)+'-'+str(j) +'.h5', 'w') as hf:
        hf.create_dataset('features_h5', data=features)

#-- (VGG19) --#
# variable size 
# image_data_temp   : 10    ->  144 B  ; 20 -> wrong 
# image_data        : 31    ->  18  MB ; 62 -> 36 MB
# features          : 31    ->  3   MB ; 62 -> 6  MB
# image_data-> 580 KB per image
# features  -> 100 KB per image

#-- (InceptionV3) --#
# variable size 
# features  -> 200 KB per image

#-- (ResNet50) --#
# variable size 
# features  -> 8 KB per image