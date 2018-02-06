import numpy as np
import cv2


class UIColor:
	def __init__(self, img_size, width, scale):
		self.img_size = img_size
		self.scale = float(scale)
		self.mask = np.zeros((img_size, img_size, 3), np.uint8)
		self.width = width


	def update(self, origin, points, color):
		num_pnts = len(points)
		w = int(max(1, self.width / self.scale))
		c = (color.red(), color.green(), color.blue())
		pnt1 = (int(points[num_pnts-2].x() / self.scale), int(points[num_pnts-2].y() / self.scale))
		pnt2 = (int(points[num_pnts-1].x() / self.scale), int(points[num_pnts-1].y() / self.scale))
		cv2.line(self.mask, pnt1, pnt2, c, w)
		
		hsv = cv2.cvtColor(origin, cv2.COLOR_BGR2HSV)
		hsvmask = cv2.cvtColor(self.mask, cv2.COLOR_BGR2HSV)
		width = self.img_size
		height = self.img_size
		for i in range(height):
			for j in range(width):
				if hsvmask[i][j][2]>20:
					hsv[i][j][0] = hsvmask[i][j][0]
		return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

	def update_width(self, d):
		self.width = min(100, max(20, self.width+ d))
		return self.width

	def reset(self):
		self.mask = np.zeros((self.img_size, self.img_size, 3), np.uint8)
