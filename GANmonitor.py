import numpy as np
import tensorflow as tf
from tensorflow import keras

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=10, latent_dim=128, num_classes=10, img_height=28, img_width=28, img_channels = 1, name="mnist"):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        arr = np.zeros((self.num_img,self.num_classes))
        for i in range(self.num_img):
            arr[i][i] = 1
        random_latent_vectors = tf.concat([random_latent_vectors,arr], axis = 1)
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5
        
        img = []
        for i in range(self.num_img):
            numpy_img = generated_images[i].numpy()
            img.append(numpy_img)
        img = np.array(img)
        img = img.reshape((self.img_height*self.num_img, self.img_width, self.img_channels))
        img = keras.preprocessing.image.array_to_img(img)
        img.save(f"images/{self.name}_generated_img_{epoch}.png")