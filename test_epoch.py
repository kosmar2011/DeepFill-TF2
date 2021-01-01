
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd

from utils import *
from net import *
from config import Config

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)

FLAGS = Config('./inpaint.yml')

generator = GeneratorMultiColumn()
discriminator = Discriminator()

test_dataset = tf.data.Dataset.list_files("../TEST/*.jpg")
test_dataset = test_dataset.map(load_image_train)
test_dataset = test_dataset.batch(FLAGS.batch_size)
test_dataset = test_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

checkpoint_dir = "./training_checkpoints"
checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
checkpoint.restore(checkpoint_dir+'/'+'ckpt-81')
step = np.int(checkpoint.step)
print("Continue Training from epoch ", step)

#restore CSV
df_load = pd.read_csv(f'./CSV_loss/loss_{step}.csv', delimiter=',')
g_total = df_load['g_total'].values.tolist()
g_total = CSV_reader(g_total)
g_hinge = df_load['g_hinge'].values.tolist()
g_hinge = CSV_reader(g_hinge)
g_l1 = df_load['g_l1'].values.tolist()
g_l1 = CSV_reader(g_l1)
d = df_load['d'].values.tolist()
d = CSV_reader(d)
print(f'Loaded CSV for step: {step}')

for data in test_dataset.take(15):
  generate_images(data, generator, training=False, num_epoch=step)

plot_history(g_total, g_hinge, g_l1, d, step, training=False)
