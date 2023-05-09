import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model

class tabularWGANGP(Model):
    ''' 
    Conditional GAN that builds samples from the label
    While GANs are more commonly used on images, this one is trained on tabular data
    '''
    def __init__(self, generator, discriminator, num_classes = 2, latent_dim = 32, out_shape = 10, d_steps = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_shape = out_shape 
        self.num_classes = num_classes
        self.d_steps = d_steps
        self.gp_weight = 10
        
        self.generator = generator
        self.discriminator = discriminator
        
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")
        
    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]
    
    def compile(self, generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss):
        super().compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        
    def gradient_penalty(self, batch_size, real_samples, fake_samples, labels):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated sample
        and added to the discriminator loss.
        """
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        diff = fake_samples - real_samples
        interpolated = real_samples + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator([interpolated, labels], training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    @tf.function
    def train_step(self, batch):
        samples, labels = batch
        samples, labels = tf.cast(samples, dtype=tf.float32), tf.cast(labels, dtype=tf.float32)
        
        batch_size = tf.shape(samples)[0]
        
        # trainging the discriminator d_steps time
        for _ in range(self.d_steps):
            noise = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_samples = self.generator([noise,labels], training = True)

            with tf.GradientTape() as tape:
                real_output = self.discriminator([samples,labels], training = True)
                fake_output = self.discriminator([generated_samples,labels], training = True)

                d_loss = self.discriminator_loss(real_output, fake_output)                
                gp = self.gradient_penalty(batch_size, samples, generated_samples, labels)
                d_loss = d_loss + gp * self.gp_weight

            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.discriminator_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )
        
        # training the generator 1 time
        noise = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            generated_samples = self.generator([noise,labels], training = True)
            predictions = self.discriminator([generated_samples,labels], training = True)
            g_loss = self.generator_loss(predictions)
            
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result()
        }