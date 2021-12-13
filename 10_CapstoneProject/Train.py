import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow import keras

seed = 232
np.random.seed(seed)
tf.random.set_seed(seed)

learning_rate = 0.001


def VGG(learning_rate):
    
    base_model = VGG16(
        weights='imagenet',
        include_top=False, 
        input_shape=(150, 150, 3)
    )
    
    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(150, 150, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    outputs = keras.layers.Dense(1)(vectors)
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate)
    loss = keras.losses.BinaryCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

# ### Model Tuning - Data Augmentation

print("Pre-processing...")

# Preprocessing the training set
train_gen = ImageDataGenerator(rescale = 1./255, 
                                rotation_range=15,
                                zoom_range=0.1,
                                shear_range=10,
                                horizontal_flip=True)

aug_train_ds = train_gen.flow_from_directory('./Data/training',
                                                 target_size = (150, 150),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the validation set
val_gen = ImageDataGenerator(rescale = 1./255)
valid_ds = val_gen.flow_from_directory('./Data/validation',
                                            target_size = (150, 150),
                                            batch_size = 32,
                                            class_mode = 'binary')


# In[381]:

print("Model Training...")

chechpoint = keras.callbacks.ModelCheckpoint(
    'VGG16_v3_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

model = VGG(learning_rate)

history = model.fit(
    aug_train_ds,
    epochs=50,
    validation_data=valid_ds,
    callbacks=[chechpoint]
)

print("Training completed!")
print("Model saved used check-pointing.")