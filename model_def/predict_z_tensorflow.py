import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import cv2

from model_def.ops import *
from model_def.utils import *

def conv_out_size_same(size, stride):
	return int(math.ceil(float(size) / float(stride)))

class PREDICT_Z(object):
	def __init__(self, sess, dcgan, dcgan_sess, input_height=64, input_width=64, crop=True,
				 batch_size=64, sample_num = 64, output_height=64, output_width=64,
				 z_dim=128, pf_dim=128, c_dim=3, dataset_name='default',
				 input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
		"""

		Symmetric to generator

		"""
		self.sess = sess
		self.dcgan = dcgan
		self.dcgan_sess = dcgan_sess
		self.crop = crop

		self.batch_size = batch_size
		self.sample_num = sample_num

		self.input_height = input_height
		self.input_width = input_width
		self.output_height = output_height
		self.output_width = output_width

		self.z_dim = z_dim
		self.pf_dim = pf_dim

		# batch normalization : deals with poor initialization helps gradient flow
		self.p_bn0 = batch_norm(name='p_bn0')
		self.p_bn1 = batch_norm(name='p_bn1')
		self.p_bn2 = batch_norm(name='p_bn2')
		self.p_bn3 = batch_norm(name='p_bn3')

		self.dataset_name = dataset_name
		self.input_fname_pattern = input_fname_pattern
		self.checkpoint_dir = checkpoint_dir
		self.c_dim = c_dim
		self.z_init = None

		self.build_model()

	def build_model(self):
		if self.crop:
			image_dims = [self.output_height, self.output_width, self.c_dim]
		else:
			image_dims = [self.input_height, self.input_width, self.c_dim]

		self.targetZ = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='target_z')
		self.target = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='target_image')
		self.originZ = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='origin_z')

		self.prediction = self.predictor(self.target, self.targetZ)
		self.sampler = self.sampler(self.target, self.targetZ)

		self.p_loss = tf.reduce_mean(tf.abs(self.targetZ - self.prediction))
		self.p_loss_single = tf.reduce_mean(tf.abs(self.targetZ - self.originZ), axis = 1)

		t_vars = tf.trainable_variables()
		self.p_vars = [var for var in t_vars if 'p_' in var.name]
		
		self.saver = tf.train.Saver()

	def train(self, config):
		p_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.p_loss, var_list=self.p_vars)
		tf.global_variables_initializer().run(session=self.sess)

		counter = 1
		start_time = time.time()
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			counter = checkpoint_counter
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")

		for epoch in xrange(config.epoch):
			z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, self.dcgan.z_dim))
			samples = self.dcgan_sess.run(self.dcgan.sampler, feed_dict={self.dcgan.z: z_sample})

			# Update P network
			self.sess.run(p_optim, feed_dict={ self.target: samples , self.targetZ : z_sample })

			errP = self.p_loss.eval({ self.target: samples , self.targetZ : z_sample }, session = self.sess)

			counter += 1
			print("Epoch: [%2d] time: %4.4f, p_loss: %.8f" % (epoch, time.time() - start_time, errP))

			if np.mod(counter, 100) == 1:
				outputs = self.sess.run(self.sampler, feed_dict={ self.target: samples , self.targetZ : z_sample })
				samples2 = self.dcgan_sess.run(self.dcgan.sampler, feed_dict={self.dcgan.z: outputs})
				save_images(samples, image_manifold_size(samples.shape[0]),
							'./{}/trainz_{:02d}_{:04d}_1.png'.format(config.sample_dir, int(epoch/1000), epoch%1000))
				save_images(samples2, image_manifold_size(outputs.shape[0]),
							'./{}/trainz_{:02d}_{:04d}_2.png'.format(config.sample_dir, int(epoch/1000), epoch%1000))

			if np.mod(counter, 500) == 2:
				self.save(self.checkpoint_dir, counter)

	def initialize(self, z0):
		self.z_init = z0

	def invert(self, constraints):
		mask = cv2.bitwise_not(constraints[1])
		const = constraints[0]

		batch_size = self.z_init.shape[0]
		z_sample = self.z_init 
		samples = self.dcgan_sess.run(self.dcgan.sampler, feed_dict={self.dcgan.z: z_sample})
		samples = ((samples+1)*127.5).astype(np.uint8)
		
		#save_images(samples, image_manifold_size(samples.shape[0]), "x1.png")
		
		#here I just stack the colored stroke directly on the original image
		#uncomment 'save_images' to see what's happenning.
		for i in range(batch_size):
			samples[i] = cv2.bitwise_or(samples[i], const, mask=mask)
			samples[i] = cv2.bitwise_or(samples[i], const)
		
		#save_images(samples, image_manifold_size(samples.shape[0]), "x2.png")
		
		samples = samples.astype(np.float32)/127.5 - 1.0

		outputs = self.sess.run(self.sampler, feed_dict={ self.target: samples , self.targetZ : z_sample })
		samples2 = self.dcgan_sess.run(self.dcgan.sampler, feed_dict={self.dcgan.z: outputs})
		errP = self.p_loss_single.eval({ self.targetZ : outputs, self.originZ : self.z_init }, session = self.sess)
		
		#print (z.shape)
		#print (samples.shape)
		#print (outputs.shape)
		#print (samples2.shape)
		#print (errP.shape)
		#save_images((((samples2+1)*127.5).astype(np.uint8)), image_manifold_size(samples2.shape[0]), "x3.png")
		
		return ((samples2+1)*127.5).astype(np.uint8), outputs, errP
		
	def gen_samples(self, z0):
		size = z0.shape[0]
		z1 = np.zeros((self.batch_size - size,self.z_dim), np.float32)
		z0 = np.concatenate((z0,z1),axis=0)
		samples = self.dcgan_sess.run(self.dcgan.sampler, feed_dict={self.dcgan.z: z0})
		return ((samples[0:size]+1)*127.5).astype(np.uint8)
		

	def predictor(self, target, targetz):
		with tf.variable_scope("predictor") as scope:

			h0 = lrelu(self.p_bn0(conv2d(target, self.pf_dim, name='p_h0_conv')))
			h1 = lrelu(self.p_bn1(conv2d(h0, self.pf_dim*2, name='p_h1_conv')))
			h2 = lrelu(self.p_bn2(conv2d(h1, self.pf_dim*4, name='p_h2_conv')))
			h3 = lrelu(self.p_bn3(conv2d(h2, self.pf_dim*8, name='p_h3_conv')))
			h4 = linear(tf.reshape(h3, [self.batch_size, -1]), self.z_dim, 'p_h4_lin')

			return h4
			
	def sampler(self, target, targetz):
		with tf.variable_scope("predictor") as scope:
			scope.reuse_variables()

			h0 = lrelu(self.p_bn0(conv2d(target, self.pf_dim, name='p_h0_conv')))
			h1 = lrelu(self.p_bn1(conv2d(h0, self.pf_dim*2, name='p_h1_conv')))
			h2 = lrelu(self.p_bn2(conv2d(h1, self.pf_dim*4, name='p_h2_conv')))
			h3 = lrelu(self.p_bn3(conv2d(h2, self.pf_dim*8, name='p_h3_conv')))
			h4 = linear(tf.reshape(h3, [self.batch_size, -1]), self.z_dim, 'p_h4_lin')

			return h4

	@property
	def model_dir(self):
		return "{}_{}_{}_{}".format(
				self.dataset_name, self.batch_size,
				self.output_height, self.output_width)
			
	def save(self, checkpoint_dir, step):
		model_name = "PREDICT_Z.model"
		checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)

		self.saver.save(self.sess,
						os.path.join(checkpoint_dir, model_name),
						global_step=step)

	def load(self, checkpoint_dir):
		import re
		print(" [*] Reading checkpoints...")

		ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
			counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
			print(" [*] Success to read {}".format(ckpt_name))
			return True, counter
		else:
			print(" [*] Failed to find a checkpoint")
			return False, 0
