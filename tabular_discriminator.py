import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model

def build_discriminator(out_shape = 14, num_classes = 2, layers_dim=[512,256]):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    samples = layers.Input(shape=(out_shape,))
    labels = layers.Input(shape=(num_classes,))
    
    x = layers.concatenate([samples, labels])
    
    def dense_block(x, units, apply_dropout=False):
        x = layers.Dense(units, kernel_initializer=initializer)(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        if apply_dropout:
            x = layers.Dropout(0.3)(x)
        return x
    
    for layer_dim in layers_dim:
        x = dense_block(x,layer_dim)

    outputs = layers.Dense(1, activation='linear')(x)

    return Model(inputs = [samples,labels], outputs=outputs, name = "discriminator")