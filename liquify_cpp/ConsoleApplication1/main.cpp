#include "all.h"

IplImage* imgc;
IplImage* img1c;

bool mouse_down=false;
#define POINT_ARRAY_MAX 12
#define SPEED_LIMIT 5
int last_points_x[POINT_ARRAY_MAX], last_points_y[POINT_ARRAY_MAX], last_points_p=0;
int current_x=0, current_y=0, last_x=0, last_y=0;
int width,height,channels;
int brush_size = 50;
int brush_size_plus = brush_size*2;
int brush_strength = 60;

void liquipoint(int radius, int strength, int last_x, int last_y, int current_x, int current_y, double point_x, double point_y, double *src_x, double *src_y)
{
	double rs = radius*radius;
	double xc =(point_x - last_x)*(point_x - last_x) + (point_y - last_y)*(point_y - last_y);
	if (rs < xc)
	{
		*src_x = point_x;
		*src_y = point_y;
	}
	else
	{
		double tb = 1 - xc / rs;
		double tc = strength/100.0 * tb;
		*src_x = point_x - tc * (current_x - last_x);
		*src_y = point_y - tc * (current_y - last_y);
	}
}

void OnMouseMove()
{
	BwImage img(imgc);
	BwImage img1(img1c);

	for (int i = last_y - brush_size_plus; i < last_y + brush_size_plus; i++)
	for (int j = last_x - brush_size_plus; j < last_x + brush_size_plus; j++) {
		if (i >= 0 && i < height && j >= 0 && j < width) {
			double src_x = j;
			double src_y = i;
			for (int k = last_points_p - 1; k >= 1; k--) {
				double jt = src_x;
				double it = src_y;
				int xs = last_points_x[k];
				int ys = last_points_y[k];
				int last_xs = last_points_x[k-1];
				int last_ys = last_points_y[k-1];
				liquipoint(brush_size, brush_strength, last_xs, last_ys, xs, ys, jt, it, &src_x, &src_y);
			}
			if (src_x >= 0 && src_x < width-1 && src_y >= 0 && src_y < height-1) {
				int x1 = src_x;
				int y1 = src_y;
				double t = src_x - x1;
				double u = src_y - y1;

				unsigned char b1 = img[y1][x1*3+0];
				unsigned char g1 = img[y1][x1*3+1];
				unsigned char r1 = img[y1][x1*3+2];
				unsigned char b2 = img[y1][x1*3+3];
				unsigned char g2 = img[y1][x1*3+4];
				unsigned char r2 = img[y1][x1*3+5];
				unsigned char b3 = img[y1+1][x1*3+0];
				unsigned char g3 = img[y1+1][x1*3+1];
				unsigned char r3 = img[y1+1][x1*3+2];
				unsigned char b4 = img[y1+1][x1*3+3];
				unsigned char g4 = img[y1+1][x1*3+4];
				unsigned char r4 = img[y1+1][x1*3+5];

				img1[i][j*3+0] = (1 - t)*(1 - u)*b1 + t*(1 - u)*b2 + (1 - t)*u*b3 + t*u*b4;
				img1[i][j*3+1] = (1 - t)*(1 - u)*g1 + t*(1 - u)*g2 + (1 - t)*u*g3 + t*u*g4;
				img1[i][j*3+2] = (1 - t)*(1 - u)*r1 + t*(1 - u)*r2 + (1 - t)*u*r3 + t*u*r4;
			}
			else {
				img1[i][j*3+0] = 0;
				img1[i][j*3+1] = 0;
				img1[i][j*3+2] = 0;
			}
		}
	}
}

void mouse(int mouseevent, int x, int y, int flags, void* param)
{
	cvCopyImage(imgc,img1c);
	if (mouseevent == CV_EVENT_LBUTTONDOWN) {
		last_points_x[0] = x;
		last_points_y[0] = y;
		last_points_p=1;
		current_x=x;
		current_y=y;
		mouse_down = true;
	}
	if (mouseevent == CV_EVENT_MOUSEMOVE) {
		if (mouse_down) {
			//limit cursor speed
			double speed =sqrt((x - current_x)*(x - current_x) + (y - current_y)*(y - current_y));
			if (speed>SPEED_LIMIT) {
				x = int(current_x + (x - current_x)/speed*SPEED_LIMIT);
				y = int(current_y + (y - current_y)/speed*SPEED_LIMIT);
			}

			last_x=current_x;
			last_y=current_y;
			current_x=x;
			current_y=y;
			last_points_x[last_points_p] = x;
			last_points_y[last_points_p] = y;
			++last_points_p;
			if (last_points_p>=2) OnMouseMove();
			if (last_points_p>=POINT_ARRAY_MAX) {
				cvCopyImage(img1c,imgc);
				last_points_x[0] = x;
				last_points_y[0] = y;
				last_points_p=1;
				current_x=x;
				current_y=y;
			}
		}
		cvCircle(img1c,cvPoint(x,y),brush_size,cvScalar(0,0,0),1);
	}
	if (mouseevent == CV_EVENT_LBUTTONUP) {
		mouse_down = false;
		if (last_points_p>=2) OnMouseMove();
		cvCopyImage(img1c,imgc);
	}
}

int main()
{
	cvNamedWindow("win1");
	cvSetMouseCallback("win1", mouse);

	imgc = cvLoadImage("untitled.png");
	width = imgc->width;
	height = imgc->height;
	channels = imgc->nChannels;
	if (channels!=3) return -1;
	img1c = cvCreateImage(cvSize(width,height),8,channels);
	cvCopyImage(imgc,img1c);

	while (1) {
		cvShowImage("win1",img1c);
		if (cvWaitKey(1)==' ') break;
	}

	return 0;
}