import os
import cv2
from PyQt5.QtWidgets import QFileDialog

def save_image(image):
	save_dir = QFileDialog.getSaveFileName(None, 'Select a folder to save the image', '.', 'JPG(*.jpg);;PNG (*.png);;BMP (*.bmp)')
	save_dir = str(save_dir[0])
	print('save to (%s)' % save_dir)
	
	if image is not None:
		cv2.imwrite(save_dir, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def load_image():
	load_dir = QFileDialog.getOpenFileName(None, 'Select an image to load', '.')
	load_dir = str(load_dir[0])
	print('load from (%s)' % load_dir)
	
	image = cv2.imread(load_dir)
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

