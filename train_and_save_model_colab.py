# Run this file in Google Colab (Runtime -> Change runtime type -> GPU)
# It trains a CNN on MNIST and saves 'digit_recognition_model.keras' into /model
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
from tensorflow.keras.datasets import mnist

os.makedirs('model', exist_ok=True)
MODEL_PATH = os.path.join('model','digit_recognition_model.keras')

(x_train, y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.0
x_test  = x_test.reshape(-1,28,28,1).astype('float32')/255.0

model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    layers.BatchNormalization(),
    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(256,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10,activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=12,batch_size=128,validation_split=0.1,verbose=2)
loss, acc = model.evaluate(x_test,y_test,verbose=2)
print(f"Test accuracy: {acc*100:.2f}%")
model.save(MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
