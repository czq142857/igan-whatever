#ifndef ALL_H
#define ALL_H

#include "highgui.h"
#include <cv.h>
#include <iostream>
#include <fstream>
#include <cmath>
#define pi 3.1415926536
using namespace cv;

class BwImage {
public:
	unsigned char *imageData;
	int widthp;
	BwImage() {}
	BwImage(IplImage* img) {imageData=(unsigned char *)(img->imageData);widthp=img->widthStep;}
	~BwImage(){}
	void init(IplImage* img) {imageData=(unsigned char *)(img->imageData);widthp=img->widthStep;}
	inline unsigned char* operator[](int rowIndx) {return (imageData + rowIndx*widthp);}
};

#endif