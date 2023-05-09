import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model

def build_generator(latent_dim = 32, out_shape = 14, num_classes = 2, layers_dim=[256,512]):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    samples = layers.Input(shape=(latent_dim,))
    labels = layers.Input(shape=(num_classes,))
    
    x = layers.concatenate([samples, labels])

    def dense_block(x, units, apply_droupout = True):
        x = layers.Dense(units)(x)        
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        if (apply_droupout):
            x = layers.Dropout(0.2)(x)
        return x
        
    for layer_dim in layers_dim:
        x = dense_block(x,layer_dim)    

    outputs = layers.Dense(out_shape, activation='tanh')(x)

    return Model(inputs = [samples,labels], outputs=outputs, name = "generator")