from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from os import listdir
from os.path import isfile, join

img_width, img_height = 150, 150

train_data_dir = 'C:/Users/Dushyant Singh/Desktop/Train'
validation_data_dir = 'C:/Users/Dushyant Singh/Desktop/Validation'
nb_train_samples = 120
nb_validation_samples = 30
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


import keras
from keras import optimizers
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.0001),
              metrics=['accuracy'])



# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

print(train_generator.class_indices)

img, labels = next(train_generator)

from skimage import io

def imshow(image_RGB):
    io.imshow(image_RGB)
    io.show()
    
import matplotlib.pyplot as plt
image_batch, label_batch = train_generator.next()

print(len(image_batch))
for i in range(0,len(image_batch)):
    image=image_batch[i]
    print(label_batch[i])
    imshow(image)

    

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

history=model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)



import matplotlib.pyplot as plt

print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')
plt.show()


predict_dir_path='C:/Users/Dushyant Singh/Desktop/Test/'

onlyfiles=[f for f in listdir(predict_dir_path)
           if isfile(join(predict_dir_path,f))]

print(onlyfiles)


from keras.preprocessing import image

dog_counter=0
cat_counter=0
for file in onlyfiles:
    img = image.load_img(predict_dir_path + file, target_size=(img_width, img_height))
    x= image.img_to_array(img)
    x=np.expand_dims(x,axis=0)

    images=np.vstack([x])
    classes=model.predict_classes(images,batch_size=10)
    classes=classes[0][0]

    if classes==0:
        print(file+": "+ 'cat')
        cat_counter+=1

    else:
        print(file+": "+'dog')
        dog_counter+=1


print("Total Dogs :", dog_counter)
print("Total Cats :", cat_counter)











                              

