import numpy as np
import time
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from lib import utils
# from scipy import ndimage
from .ui_brush import UIBrush
from .ui_color import UIColor
from .ui_liquify import UILiquify

class GUIDraw(QWidget):

	update_image_id = pyqtSignal(int)
	update_frame_id = pyqtSignal(int)
	update_color = pyqtSignal(str)

	def __init__(self, opt_engine, win_size=320, img_size=64, topK=16):
		QWidget.__init__(self)
		self.isPressed = False
		self.points = []
		self.topK = topK
		self.lastDraw = 0
		self.model = None
		self.init_color()
		self.opt_engine = opt_engine
		self.pos = None
		self.nps = win_size
		self.scale = win_size / float(img_size)
		self.brushWidth = int(8 * self.scale)
		self.show_nn = True
		self.type = 'brush'
		self.uiBrush = UIBrush(img_size=img_size, width=self.brushWidth, scale=self.scale)
		self.uiColor = UIColor(img_size=img_size, width=self.brushWidth, scale=self.scale)
		self.uiLiquify = UILiquify(img_size=img_size, width=self.brushWidth*2, scale=self.scale)
		self.img_size = img_size
		self.move(win_size, win_size)
		self.frame_id = 0
		self.image_id = 0
		self.num_frames = self.opt_engine.get_num_frames()
		self.num_images = self.opt_engine.get_num_images()
		self.origin_img = self.opt_engine.get_image(self.image_id, self.frame_id)
		self.current_img = np.copy(self.origin_img)
		self.setMouseTracking(True)

	def update_im(self):
		self.update()
		QApplication.processEvents()

	def update_ui(self):
		if self.type is 'brush':
			self.current_img = self.uiBrush.update(self.origin_img, self.points, self.color)
		if self.type is 'color':
			self.current_img = self.uiColor.update(self.origin_img, self.points, self.color)
		if self.type is 'liquify':
			self.current_img = self.uiLiquify.update(self.origin_img, self.points)

	def set_image_id(self, image_id):
		self.image_id = image_id
		self.origin_img = self.opt_engine.get_image(self.image_id, self.frame_id)
		self.current_img = np.copy(self.origin_img)
		self.update()

	def set_frame_id(self, frame_id):
		self.frame_id = frame_id
		self.origin_img = self.opt_engine.get_image(self.image_id, self.frame_id)
		self.current_img = np.copy(self.origin_img)
		self.update()


	def reset(self):
		self.isPressed = False
		self.points = []
		self.lastDraw = 0
		self.uiBrush.reset()
		self.uiColor.reset()
		self.uiLiquify.reset()
		self.frame_id = self.num_frames-1
		self.image_id = 0
		self.origin_img = self.opt_engine.get_image(self.image_id, self.frame_id)
		self.current_img = np.copy(self.origin_img)

		self.update()

	def round_point(self, pnt):
		# print(type(pnt))
		x = int(np.round(pnt.x()))
		y = int(np.round(pnt.y()))
		return QPoint(x, y)

	def init_color(self):
		self.color = QColor(0, 0, 255)  # default color blue
		self.prev_color = self.color

	def change_color(self):
		color = QColorDialog.getColor(parent=self)
		self.color = color
		self.prev_color = self.color
		self.update_color.emit(('background-color: %s' % self.color.name()))

	def paintEvent(self, event):
		painter = QPainter()
		painter.begin(self)
		painter.fillRect(event.rect(), Qt.white)
		painter.setRenderHint(QPainter.Antialiasing)

		bigim = cv2.resize(self.current_img, (self.nps, self.nps))
		qImg = QImage(bigim.tostring(), self.nps, self.nps, QImage.Format_RGB888)
		painter.drawImage(0, 0, qImg)

		# draw cursor
		if self.pos is not None:
			w = self.brushWidth/2
			c = self.color
			pnt = QPointF(self.pos.x(), self.pos.y())
			if self.type is 'brush':
				ca = QColor(c.red(), c.green(), c.blue(), 127)
			if self.type is 'color':
				ca = QColor(c.red(), c.green(), c.blue(), 127)
			if self.type is 'liquify':
				ca = QColor(0, 0, 0, 255)

			painter.setPen(QPen(ca, 1))
			if self.type is not 'liquify':
				painter.setBrush(ca)
			painter.drawEllipse(pnt, w, w)

		painter.end()

	def wheelEvent(self, event):
		d = event.angleDelta().y() / 100
		if self.type is 'brush':
			self.brushWidth = self.uiBrush.update_width(d)
		if self.type is 'color':
			self.brushWidth = self.uiColor.update_width(d)
		if self.type is 'liquify':
			self.brushWidth = self.uiLiquify.update_width(d)
		self.update()

	def mousePressEvent(self, event):
		self.pos = self.round_point(event.pos())
		if event.button() == Qt.LeftButton:
			self.isPressed = True
			self.opt_engine.dirty = True
			self.points.append(self.pos)
			self.update_ui()
		if event.button() == Qt.RightButton:
			self.change_color()
		self.update()

	def mouseMoveEvent(self, event):
		self.pos = self.round_point(event.pos())
		if self.isPressed:
			self.points.append(self.pos)
			self.update_ui()
		self.update()

	def mouseReleaseEvent(self, event):
		self.pos = self.round_point(event.pos())
		if event.button() == Qt.LeftButton and self.isPressed:
			self.uiBrush.reset()
			self.uiColor.reset()
			self.uiLiquify.reset()
			del self.points[:]
			self.isPressed = False
			self.lastDraw = 0
			self.opt_engine.set_image(self.image_id, self.frame_id, self.current_img)
			self.origin_img = np.copy(self.current_img)
			self.update_image_id.emit(self.image_id)
		self.update()

	def update_frame(self, dif):
		self.frame_id = (self.frame_id+dif) % self.num_frames
		self.origin_img = self.opt_engine.get_image(self.image_id, self.frame_id)
		self.current_img = np.copy(self.origin_img)
		print("show frame id = %d"%self.frame_id)

	def next(self):
		self.opt_engine.next(self.image_id, self.frame_id)
		self.isPressed = False
		self.points = []
		self.lastDraw = 0
		self.uiBrush.reset()
		self.uiColor.reset()
		self.uiLiquify.reset()
		self.set_image_id(0)
		self.update_image_id.emit(0)
		self.set_frame_id(self.num_frames-1)
		self.update_frame_id.emit(self.num_frames-1)
		self.origin_img = self.opt_engine.get_image(self.image_id, self.frame_id)
		self.current_img = np.copy(self.origin_img)
		self.update()

	def undo(self):
		self.opt_engine.undo()
		self.isPressed = False
		self.points = []
		self.lastDraw = 0
		self.uiBrush.reset()
		self.uiColor.reset()
		self.uiLiquify.reset()
		self.update_image_id.emit(self.image_id)
		self.origin_img = self.opt_engine.get_image(self.image_id, self.frame_id)
		self.current_img = np.copy(self.origin_img)
		self.update()

	def morph_seq(self):
		self.frame_id=0
		print('show %d frames' % self.num_frames)
		for n in range(self.num_frames):
			self.update()
			QApplication.processEvents()
			fps = 10
			time.sleep(1/float(fps))
			self.update_frame_id.emit(self.frame_id)
			if n < self.num_frames-1: # stop at last frame
				self.update_frame(1)

	def use_brush(self):
		self.type = 'brush'
		self.color = self.prev_color
		self.update_color.emit(('background-color: %s' % self.color.name()))
		self.brushWidth = self.uiBrush.update_width(0)
		self.update()

	def use_color(self):
		self.type = 'color'
		self.color = self.prev_color
		self.update_color.emit(('background-color: %s' % self.color.name()))
		self.brushWidth = self.uiColor.update_width(0)
		self.update()

	def use_liquify(self):
		self.type = 'liquify'
		self.brushWidth = self.uiLiquify.update_width(0)
		self.update()


