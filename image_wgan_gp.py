import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model

class imageWGANGP(Model):
    def __init__(self, generator, discriminator, latent_dim=128, num_classes=10, img_height=28, img_width=28, img_channels=1, d_steps=5):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.generator = generator        
        self.discriminator = discriminator
        
        self.image_height = img_height
        self.image_width = img_width
        self.image_channels = img_channels
        
        self.gp_weight = 10
        self.d_steps = d_steps
        
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
    
    def gradient_penalty(self, batch_size, real_images, fake_images, label_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator([interpolated, label_images], training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp
    
    @tf.function
    def train_step(self, batch):
        real_images, labels = batch
        
        # turning labels into images with num_classes channels for discriminator input
        label_images = labels[:, :, None, None]
        label_images = tf.repeat(
            label_images, repeats=[self.image_height * self.image_width]
        )
        label_images = tf.reshape(
            label_images, (-1, self.image_height, self.image_width, self.num_classes)
        )
        
        # generating noise for the generator
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        # generator also takes label as part of the input
        random_vector_labels = tf.concat(
            [random_latent_vectors, labels], axis=1
        )
        
        # training the discriminator d_steps times
        for step in range(self.d_steps):
            # generating the images
            generated_images = self.generator(random_vector_labels, training = True)
        
            with tf.GradientTape() as tape:
                real_output = self.discriminator([real_images, label_images], training = True)
                fake_output = self.discriminator([generated_images, label_images], training = True)

                d_loss = self.discriminator_loss(real_output,fake_output)
                # adding gradient penalty to discriminator loss
                gp = self.gradient_penalty(batch_size, real_images, generated_images, label_images)
                d_loss = d_loss + gp * self.gp_weight

            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.discriminator_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )
        
        # training the generator once
        # generating noise for generator training
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, labels], axis=1
        )
        
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels, training = True)
            predictions = self.discriminator([fake_images,label_images], training = True)
            g_loss = self.generator_loss(predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        
        # tracking loss values
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }