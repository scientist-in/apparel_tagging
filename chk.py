from vgg19 import VGG19
from inception_v3 import InceptionV3
from resnet50 import ResNet50
from vgg19  import VGG19

from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model
import ipdb
import numpy as np
import sys
ipdb.set_trace()
from vgg16 import VGG16
from keras.preprocessing import image
from imagenet_utils import preprocess_input

#prep image locations
image_path_file = '/home/keeda/Documents/scientist/demo/apparel_tagging/fashion-data/train.txt'
images_loc = '/home/keeda/Documents/scientist/demo/apparel_tagging/fashion-data/images/'
with open(image_path_file) as f:
    image_paths = f.readlines()
image_paths = [images_loc+x.strip('\n')+'.jpg' for x in image_paths] 

#prep model
model = VGG16(weights='imagenet', include_top=True)
model = VGG19(weights= 'imagenet',include_top = True)
model = InceptionV3(weights='imagenet', include_top=True)
model = ResNet50(weights= 'imagenet', include_top =True)

img_path = 'elephant.jpg'
img_path2 = 'cat.jpg'

#prep first image
img = image.load_img(image_paths[0], target_size=(224, 224))
img = image.img_to_array(img)
image_data_temp = np.expand_dims(img, axis=0)
image_data = image_data_temp
#prep rest of the image
ipdb.set_trace()
for idx,image_path in enumerate(image_paths[1:3000]):
    if idx%200==0:
        image_data = np.append(image_data,image_data_temp,axis=0)
        img = image.load_img(image_paths[idx], target_size=(224, 224))
        img = image.img_to_array(img)
        image_data_temp = np.expand_dims(img, axis=0)
        continue
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    image_data_temp = np.append(image_data_temp,img,axis=0)
    print idx
    #sys.stdout.flush()
ipdb.set_trace()
# img1 = image.load_img(img_path, target_size=(224, 224))
# img2 = image.load_img(img_path2 , target_size=(224, 224))
# x1 = image.img_to_array(img1)
# x1 = np.expand_dims(x1, axis=0)
# 
# x2 = image.img_to_array(img2)
# x2 = np.expand_dims(x2, axis=0)
# 
# x = np.append(x1,x2,axis=0)
#x = preprocess_input(x)

features = model.predict(x,batch_size = 100, verbose =1)

#np.append(x1,x2,axis=0).shape