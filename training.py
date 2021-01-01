import tensorflow as tf
import numpy as np
import pandas as pd

import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import datetime
import time

from net import *
from utils import *
from config import *


tf.random.set_seed(20)
tf.config.run_functions_eagerly(True)

FLAGS = Config('./inpaint.yml')
img_shapes = FLAGS.img_shapes

BATCH_SIZE = FLAGS.batch_size 
img_shape = FLAGS.img_shapes
IMG_HEIGHT = img_shape[0]
IMG_WIDTH = img_shape[1]

training_dirs = "./TRAIN"
validation_dirs = "./TEST"

#IMG PRE-PROCESSING
def load(img):
  img = tf.io.read_file(img)
  img = tf.image.decode_jpeg(img)
  return tf.cast(img, tf.float32)

def normalize(img):
  return (img /127.5) - 1.

def load_image_train(img):
  img = load(img)
  img = resize_pipeline(img, IMG_HEIGHT, IMG_WIDTH)
  return normalize(img)

def resize_pipeline(img, height, width):
  return tf.image.resize(img, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  

generator = Generator()
discriminator = Discriminator()

BUFFER_SIZE = 4000

train_dataset = tf.data.Dataset.list_files(training_dirs+'/*.jpg')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.cache("../../../tmp/CACHED_TRAIN.tmp")
train_dataset = train_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) 

test_dataset = tf.data.Dataset.list_files(validation_dirs +'/*.jpg')
test_dataset = test_dataset.map(load_image_train)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


def generator_loss(input, stage1, stage2, neg):
    gen_l1_loss = tf.reduce_mean(tf.abs(input - stage1)) 
    gen_l1_loss +=  tf.reduce_mean(tf.abs(input - stage2))
    gen_hinge_loss = -tf.reduce_mean(neg) 
    total_gen_loss = gen_hinge_loss + gen_l1_loss
    return total_gen_loss, gen_hinge_loss, gen_l1_loss  

def dicriminator_loss(pos, neg):
    hinge_pos = tf.reduce_mean(tf.nn.relu(1.0 - pos)) #ειναι tf.nn.relu γιατι θελουμε max(features,0) απο hinge
    hinge_neg = tf.reduce_mean(tf.nn.relu(1.0 + neg))    
    return  tf.add(.5 * hinge_pos, .5 * hinge_neg) # apo to neural gym 
    
    #cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)
    #real_loss = cross_entropy(tf.ones_like(pos), pos)
    #fake_loss = cross_entropy(tf.zeros_like(neg), neg)
    #return real_loss + fake_loss 
    
#OPTIMIZERS
#PAPER 1E-4, 0.5, 0.9
generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9) 
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
#generator_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
#discriminator_optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)


#TRAINING
@tf.function
def train_step(input, mask):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    #input = original 
    #batch_incomplete = original+mask
    #stage2 = prediction/inpainted image 

    batch_incomplete = input*(1.-mask)

    stage1, stage2, _ = generator(batch_incomplete, mask, training=True)

    batch_complete = stage2*mask + batch_incomplete*(1.-mask)
    batch_pos_neg = tf.concat([input, batch_complete], axis=0)
    if FLAGS.gan_with_mask:
        batch_pos_neg = tf.concat([batch_pos_neg, tf.tile(mask, [FLAGS.batch_size*2, 1, 1, 1])], axis=3)

    pos_neg = discriminator(batch_pos_neg, training=True)
    pos, neg = tf.split(pos_neg, 2)

    total_gen_loss, gen_hinge_loss, gen_l1_loss = generator_loss(input, stage1, stage2, neg)
    dis_loss = dicriminator_loss(pos, neg)

  generator_gradients = gen_tape.gradient(total_gen_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(dis_loss,
                                               discriminator.trainable_variables) 
  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
  return total_gen_loss, gen_hinge_loss, gen_l1_loss, dis_loss    


def fit(train_ds, epochs, test_ds):
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        df_load = pd.read_csv(f'./CSV_loss/loss_{int(checkpoint.step)}.csv', delimiter=',')
        g_total = df_load['g_total'].tolist()
        g_total = CSV_reader(g_total)
        g_hinge = df_load['g_hinge'].tolist()
        g_hinge = CSV_reader(g_hinge)
        g_l1 = df_load['g_l1'].tolist()
        g_l1 = CSV_reader(g_l1)
        d = df_load['d'].tolist()
        d = CSV_reader(d)
        print(f"Loaded CSV from step: {int(checkpoint.step)}")
    else:
        print("Initializing from scratch.")
        g_total, g_hinge, g_l1, d = [], [], [], []

    for ep in trange(epochs):
        start = time.time()

        checkpoint.step.assign_add(1)
        g_total_b, g_hinge_b, g_l1_b, d_b = 0, 0, 0, 0
        count = len(train_ds)
	# Train
        for input_image in tqdm(train_ds):
            mask = create_mask(FLAGS)
            total_gen_loss, gen_hinge_loss, gen_l1_loss, dis_loss = train_step(input_image, mask)
            g_total_b += total_gen_loss
            g_hinge_b += gen_hinge_loss
            g_l1_b += gen_l1_loss
            d_b += dis_loss
        g_total.append(g_total_b/count)
        g_hinge.append(g_hinge_b/count)
        g_l1.append(g_l1_b/count)
        d.append(d_b/count)

        check_step = int(checkpoint.step)
        plot_history(g_total, g_hinge, g_l1, d, check_step)

        dict1 = {'g_total': g_total,
                 'g_hinge': g_hinge,
                 'g_l1': g_l1,
                 'd': d}

        gt = pd.DataFrame(dict1)
        gt.to_csv(f'./CSV_loss/loss_{check_step}.csv', index=False)


        for input in test_ds.take(1):
            generate_images(input, num_epoch=check_step)
        print("Epoch: ", check_step)

        if check_step % 2 == 0:
            save_path = manager.save()
            print(f"Saved checkpoint for step {check_step}: {save_path}")

        print (f'Time taken for epoch {check_step} is {time.time()-start} sec\n')
    manager.save()

#CHECKPOINT
checkpoint_dir = "./training_checkpoints"
checkpoint = tf.train.Checkpoint(step=tf.Variable(0), 
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

checkpoint.restore(manager.latest_checkpoint)
print("Continue Training from epoch ", np.int(checkpoint.step))



#FIT
EPOCHS = 200 - np.int(checkpoint.step)+1
fit(train_dataset, EPOCHS, test_dataset)
