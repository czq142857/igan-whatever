from __future__ import print_function
import sys
import argparse
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from ui import gui_design
from pydoc import locate
from model_def.dcgan_tensorflow import DCGAN
from model_def.predict_z_tensorflow import PREDICT_Z
import constrained_opt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def parse_args():
	parser = argparse.ArgumentParser(description='iGAN: Interactive Visual Synthesis Powered by GAN')
	parser.add_argument('--model_name', dest='model_name', help='the model name', default='shoes64', type=str)
	parser.add_argument('--win_size', dest='win_size', help='the size of the main window', type=int, default=384)
	parser.add_argument('--image_size', dest='image_size', help='The size of the output images to produce', type=int, default=64)
	parser.add_argument('--batch_size', dest='batch_size', help='the number of random initializations', type=int, default=16)
	parser.add_argument('--dimz', dest='dimz', help='Size of latent vector z', type=int, default=128)
	parser.add_argument('--n_iters', dest='n_iters', help='the number of total optimization iterations', type=int, default=5)
	parser.add_argument('--top_k', dest='top_k', help='the number of the thumbnail results being displayed', type=int, default=16)
	parser.add_argument('--morph_steps', dest='morph_steps', help='the number of intermediate frames of morphing sequence', type=int, default=16)
	parser.add_argument('--checkpoint', dest='checkpoint', help='the file that stores the generative model', type=str, default=None)
	#parser.add_argument('--d_weight', dest='d_weight', help='captures the visual realism based on GAN discriminator', type=float, default=0.0)
	parser.add_argument('--interp', dest='interp', help='the interpolation method (linear or slerp)', type=str, default='linear')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()
	if not args.checkpoint:  #if the model_file is not specified
		args.checkpoint = "models/%s/checkpoint" % args.model_name

	for arg in vars(args):
		print('[%s] =' % arg, getattr(args, arg))

	args.win_size = int(args.win_size / 4.0) * 4  # make sure the width of the image can be divided by 4

	# initialize model and constrained optimization problem

	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
	run_config = tf.ConfigProto()
	run_config.gpu_options.allow_growth=True

	sess = tf.Session(config=run_config)
	dcgan = DCGAN(
			sess,
			output_width=args.image_size,
			output_height=args.image_size,
			batch_size=args.batch_size,
			z_dim=args.dimz,
			dataset_name=args.model_name,
			checkpoint_dir=args.checkpoint)

	sess_z = tf.Session(config=run_config)
	predict_z = PREDICT_Z(
			sess_z,
			dcgan,
			sess,
			output_width=args.image_size,
			output_height=args.image_size,
			batch_size=1,
			z_dim=args.dimz,
			dataset_name=args.model_name,
			checkpoint_dir=args.checkpoint)
	if not dcgan.load(args.checkpoint)[0]:
		raise Exception("[!] Train a model first")
	if not predict_z.load(args.checkpoint + "z")[0]:
		raise Exception("[!] Train a model first")

	opt_engine = constrained_opt.Constrained_OPT(dcgan, predict_z, image_size=args.image_size, batch_size=args.batch_size, dimz=args.dimz, n_iters=args.n_iters, topK=args.top_k,
												 morph_steps=args.morph_steps, interp=args.interp)
												 
	# initialize application
	app = QApplication(sys.argv)
	window = gui_design.GUIDesign(opt_engine, win_size=args.win_size, img_size=args.image_size, topK=args.top_k, model_name=args.model_name)
	#app.setStyleSheet(qdarkstyle.load_stylesheet(pyside=False))  # comment this if you do not like dark stylesheet
	app.setWindowIcon(QIcon('logo.png'))  # load logo
	window.setWindowTitle('Interactive GAN')
	window.setWindowFlags(window.windowFlags() & ~Qt.WindowMaximizeButtonHint)   # fix window siz
	window.show()
	app.exec_()