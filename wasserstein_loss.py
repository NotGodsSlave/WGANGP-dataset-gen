import tensorflow as tf

# define wasserstein loss here
def wasserstein_discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(real_output)
    fake_loss = tf.reduce_mean(fake_output)
    return fake_loss - real_loss

def wasserstein_generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)