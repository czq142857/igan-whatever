import numpy as np
import cv2


class UIBrush:
	def __init__(self, img_size, width, scale):
		self.img_size = img_size
		self.scale = float(scale)
		self.width = width


	def update(self, origin, points, color):
		img = np.copy(origin)
		num_pnts = len(points)
		w = int(max(1, self.width / self.scale))
		c = (color.red(), color.green(), color.blue())
		for i in range(0, num_pnts - 1):
			pnt1 = (int(points[i].x() / self.scale), int(points[i].y() / self.scale))
			pnt2 = (int(points[i + 1].x() / self.scale), int(points[i + 1].y() / self.scale))
			cv2.line(img, pnt1, pnt2, c, w)
		return img

	def update_width(self, d):
		self.width = min(100, max(1, self.width+ d))
		return self.width

	def reset(self):
		return
