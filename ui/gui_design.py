from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from . import gui_draw
from . import gui_vis
import time

class GUIDesign(QWidget):
	def __init__(self, opt_engine, win_size=320, img_size=64, topK=16, model_name='tmp'):
		# draw the layout
		QWidget.__init__(self)
		morph_steps = 16
		self.opt_engine = opt_engine
		self.drawWidget = gui_draw.GUIDraw(opt_engine, win_size=win_size, img_size=img_size, topK=topK)
		self.drawWidget.setFixedSize(win_size, win_size)
		vbox = QVBoxLayout()

		self.drawWidgetBox = QGroupBox()
		self.drawWidgetBox.setTitle('Drawing Pad')
		vbox_t = QVBoxLayout()
		vbox_t.addWidget(self.drawWidget)
		self.drawWidgetBox.setLayout(vbox_t)
		vbox.addWidget(self.drawWidgetBox)
		self.slider = QSlider(Qt.Horizontal)
		vbox.addStretch(1)
		self.slider.setMinimum(0)
		self.slider.setMaximum(morph_steps-1)
		self.slider.setValue(0)
		self.slider.setTickPosition(QSlider.TicksBelow)
		self.slider.setTickInterval(1)

		vbox.addWidget(self.slider)
		vbox.addStretch(1)
		self.bBrush= QRadioButton("Brush")
		self.bBrush.setToolTip('Change the color of a specific region')
		self.bColor= QRadioButton("Color")
		self.bColor.setToolTip('Change the color while preserving lightness')
		self.bLiquify = QRadioButton("Liquify")
		self.bLiquify.setToolTip('Warp a specific region')
		self.bBrush.toggle()
		bhbox =  QHBoxLayout()
		bGroup = QButtonGroup(self)
		bGroup.addButton(self.bBrush)
		bGroup.addButton(self.bColor)
		bGroup.addButton(self.bLiquify)
		bhbox.addWidget(self.bBrush)
		bhbox.addWidget(self.bColor)
		bhbox.addWidget(self.bLiquify)

		self.colorPush  = QPushButton()  # to visualize the selected color
		self.colorPush.setFixedWidth(20)
		self.colorPush.setFixedHeight(20)
		self.colorPush.setStyleSheet("background-color: blue")
		bhbox.addWidget(self.colorPush)


		vbox.addLayout(bhbox)

		hbox = QHBoxLayout()
		hbox.addLayout(vbox)
		self.visWidgetBox = QGroupBox()
		self.visWidgetBox.setTitle('Candidate Results')
		vbox_t = QVBoxLayout()
		self.visWidget = gui_vis.GUI_VIS(opt_engine=opt_engine, grid_size=None, topK=topK, nps=win_size, model_name=model_name)
		vbox_t.addWidget(self.visWidget)
		self.visWidgetBox.setLayout(vbox_t)
		vbox2 = QVBoxLayout()

		vbox2.addWidget(self.visWidgetBox)
		vbox2.addStretch(1)

		self.bPlay = QPushButton('&Play')
		self.bPlay.setToolTip('Play a morphing sequence between the previous result and the current result')
		self.bUndo = QPushButton('&Undo')
		self.bUndo.setToolTip('Undo all modifications in the current stage')


		self.bReset = QPushButton("&Reset")
		self.bReset.setToolTip('Restore the system')
		self.bNext = QPushButton("&Next")
		self.bNext.setToolTip('Fix current result and add noise')

		chbox = QHBoxLayout()
		chbox.addWidget(self.bNext)
		chbox.addWidget(self.bPlay)
		chbox.addWidget(self.bUndo)
		chbox.addWidget(self.bReset)


		vbox2.addLayout(chbox)

		hbox.addLayout(vbox2)
		self.setLayout(hbox)
		mainWidth = self.visWidget.winWidth + win_size + 60
		mainHeight = self.visWidget.winHeight + 100

		self.setGeometry(0, 0, mainWidth, mainHeight)
		self.setFixedSize(self.width(), self.height()) # fix window size
		# connect signals and slots
		self.opt_engine.update_image.connect(self.drawWidget.update_im)
		self.opt_engine.update_image.connect(self.visWidget.update_vis)
		self.visWidget.update_image_id.connect(self.drawWidget.set_image_id)
		self.drawWidget.update_image_id.connect(self.visWidget.set_image_id)
		self.slider.valueChanged.connect(self.visWidget.set_frame_id)
		self.slider.valueChanged.connect(self.drawWidget.set_frame_id)
		self.drawWidget.update_frame_id.connect(self.visWidget.set_frame_id)
		self.drawWidget.update_frame_id.connect(self.slider.setValue)
		self.opt_engine.start()
		self.drawWidget.update()
		self.visWidget.update()
		self.bBrush.toggled.connect(self.drawWidget.use_brush)
		self.bColor.toggled.connect(self.drawWidget.use_color)
		self.bLiquify.toggled.connect(self.drawWidget.use_liquify)
		self.colorPush.clicked.connect(self.drawWidget.change_color)
		self.drawWidget.update_color.connect(self.colorPush.setStyleSheet)
		self.bPlay.clicked.connect(self.play)
		self.bUndo.clicked.connect(self.undo)
		self.bReset.clicked.connect(self.reset)
		self.bNext.clicked.connect(self.next)
		self.start_t = time.time()

	def reset(self):
		self.start_t = time.time()
		self.opt_engine.reset()
		self.drawWidget.reset()
		self.visWidget.reset()
		self.update()

	def play(self):
		self.drawWidget.morph_seq()

	def undo(self):
		self.drawWidget.undo()

	def next(self):
		print('time spent = %3.3f' % (time.time()-self.start_t))
		self.drawWidget.next()

	def keyPressEvent(self, event):
		if event.key() == Qt.Key_R:
		   self.reset()

		if event.key() == Qt.Key_Q:
			print('time spent = %3.3f' % (time.time()-self.start_t))
			self.close()

		if event.key() == Qt.Key_U:
			self.undo()

		if event.key() == Qt.Key_P:
			self.play()

		if event.key() == Qt.Key_N:
			self.next()