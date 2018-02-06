from __future__ import print_function
from time import time
import numpy as np
import sys
from lib import utils
from PyQt5.QtCore import *
import cv2

class Constrained_OPT(QThread):

	update_image = pyqtSignal()

	def __init__(self, model, opt_solver, image_size=64, batch_size=16, dimz = 128, n_iters=1, topK=16, morph_steps=16, interp='linear'):
		QThread.__init__(self)
		self.model = model #GAN network, 'dcgan'
		self.opt_solver = opt_solver #Encoder network, 'predict_z'
		self.topK = topK
		self.max_iters = n_iters
		self.fixed_iters = 150  # [hack] after 150 iterations, do not change the order of the results
		self.image_size = image_size
		self.batch_size = batch_size
		self.dimz = dimz
		self.morph_steps = morph_steps  # number of intermediate frames
		self.interp = interp  # interpolation method
		# data
		self.z_seq = None	 # sequence of latent vector
		self.img_seq = np.full((self.batch_size,self.morph_steps,self.image_size,self.image_size,3), 255, np.uint8)   # sequence of images
		self.img_seq_backup = np.full((self.batch_size,self.morph_steps,self.image_size,self.image_size,3), 255, np.uint8)   # sequence of images backup
		self.prev_z = None  # previous latent vector
		self.prev_img = np.full((1,self.image_size,self.image_size,3), 255, np.uint8)
		# current frames
		self.current_ims = np.full((1,self.image_size,self.image_size,3), 255, np.uint8)   # the images being displayed now
		self.order = range(self.topK)
		self.dirty = False #canvas is modified -> dirty
		self.z_init = None
		self.prev_zs = None
		self.current_zs = None
		self.init_z()			# initialize latent vectors

	def init_z(self, frame_id=-1, image_id=-1):
		n_sigma = 0.5
		# set prev_z
		if self.z_seq is not None and image_id >= 0:
			image_id = image_id % self.z_seq.shape[0]
			frame_id = frame_id % self.z_seq.shape[1]
			print('set z as image %d, frame %d' % (image_id, frame_id))
			self.prev_z = self.z_seq[image_id, frame_id]

		if self.prev_z is None:  #random initialization
			self.z_init = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.dimz))
			self.prev_zs = self.z_init
		else:  # add small noise to initial latent vector, so that we can get different results
			z0_r = np.tile(self.prev_z, [self.batch_size, 1])
			z0_n = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.dimz)) * n_sigma
			self.z_init = np.clip(z0_r + z0_n, -0.99, 0.99)
			self.prev_zs = z0_r

	def next(self, image_id, frame_id, n_sigma = 0.4):
		print('next step', frame_id, image_id)
		if self.dirty:
			self.next_modified(image_id,frame_id,self.morph_steps,n_sigma)
		else:
			self.next_unchanged(image_id,frame_id,self.morph_steps,n_sigma)
		self.dirty = False
	
	def next_unchanged(self, image_id, frame_id, n_steps, n_sigma):
		# set prev_z
		if self.z_seq is not None and image_id >= 0:
			print('set z as image %d, frame %d' % (image_id, frame_id))
			self.prev_z = self.z_seq[image_id, frame_id]

		if self.prev_z is None: #random initialization
			self.z_init = np.random.uniform(-1.0, 1.0, size=(self.topK, self.dimz))
			self.prev_zs = self.z_init
			self.current_zs = self.z_init
		else:  # add small noise to initial latent vector, so that we can get different results
			z0_r = np.tile(self.prev_z, [self.topK, 1])
			z0_n = np.random.uniform(-1.0, 1.0, size=(self.topK, self.dimz)) * n_sigma
			self.z_init = z0_r
			self.prev_zs = z0_r
			self.current_zs = np.clip(z0_r + z0_n, -0.99, 0.99)
		
		#generate samples
		self.current_ims = self.opt_solver.gen_samples(self.current_zs)
		self.order = range(self.topK)
		
		#generate morphing
		self.gen_morphing(self.interp, self.morph_steps)
		self.update_image.emit()

	def next_modified(self, image_id, frame_id, n_steps, n_sigma):
		#set prev
		print('set target as image %d, frame %d' % (image_id, frame_id))
		self.prev_img = self.img_seq[image_id, frame_id]

		#generate samples
		self.update_invert(self.prev_img, n_sigma)
		
		#generate morphing
		self.gen_morphing(self.interp, self.morph_steps)
		self.update_image.emit()

	def get_image(self, image_id, frame_id):
		return self.img_seq[image_id, frame_id]

	def set_image(self, image_id, frame_id, img):
		self.img_seq[image_id, frame_id] = img

	def get_images(self, frame_id):
		return self.img_seq[:, frame_id]

	def get_num_images(self):
		return self.batch_size

	def get_num_frames(self):
		return self.morph_steps

	def get_current_results(self):
		return self.current_ims

	def run(self):  # main function
		time_to_wait = 50 #millisecond
		while (1):
			self.msleep(time_to_wait)

	def update_invert(self, targetimage, n_sigma):
		self.prev_z, self.prev_zs, gx_t, z_t, cost_all = self.opt_solver.invert(targetimage, n_sigma)

		order = np.argsort(cost_all)
		order = order[0:self.topK]

		self.order = order
		self.current_ims = gx_t[order]
		self.current_zs = z_t[order]
		self.update_image.emit()

	def gen_morphing(self, interp='linear', n_steps=16):
		if self.current_ims is None:
			return

		z1 = self.prev_zs[self.order]
		z2 = self.current_zs
		t = time()
		img_seq = []
		z_seq = []

		for n in range(n_steps):
			ratio = n / float(n_steps- 1)
			z_t = utils.interp_z(z1, z2, ratio, interp=interp)
			#generate linear interpolations
			seq = self.opt_solver.gen_samples(z_t)
			img_seq.append(seq[:, np.newaxis, ...])
			z_seq.append(z_t[:,np.newaxis,...])
		self.img_seq = np.concatenate(img_seq, axis=1)
		self.img_seq_backup = np.copy(self.img_seq)
		self.z_seq = np.concatenate(z_seq, axis=1)
		print('generate morphing sequence (%.3f seconds)' % (time()-t))

	def undo(self):
		self.img_seq = self.img_seq_backup

	def reset(self):
		self.prev_z = None
		self.z_seq = None
		self.init_z()
		self.img_seq = np.full((self.batch_size,self.morph_steps,self.image_size,self.image_size,3), 255, np.uint8)
		self.img_seq_backup = np.full((self.batch_size,self.morph_steps,self.image_size,self.image_size,3), 255, np.uint8)
		self.current_ims = np.full((1,self.image_size,self.image_size,3), 255, np.uint8)
		self.order = range(self.topK)



