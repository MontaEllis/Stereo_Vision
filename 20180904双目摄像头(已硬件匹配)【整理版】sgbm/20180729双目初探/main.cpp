//                   _ooOoo_
//                  o8888888o
//                  88" . "88
//                  (| -_- |)
//                  O\  =  /O
//               ____/`---'\____
//             .'  \\|     |//  `.
//            /  \\|||  :  |||//  \
//           /  _||||| -:- |||||-  \
//           |   | \\\  -  /// |   |
//           | \_|  ''\---/''  |   |
//           \  .-\__  `-`  ___/-. /
//         ___`. .'  /--.--\  `. . __
//      ."" '<  `.___\_<|>_/___.'  >'"".
//     | | :  `- \`.;`\ _ /`;.`/ - ` : | |
//     \  \ `-.   \_ __\ /__ _/   .-` /  /
//======`-.____`-.___\_____/___.-`____.-'======
//                   `=---='
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//         佛祖保佑       永无BUG
//  本模块已经经过开光处理，绝无可能再产生bug
//=============================================


/*******************************************************************
*  Copyright(c) :Ellis
*  All rights reserved.
*
*  文件名称:双目初探
*  简要描述:SGBM双目匹配
*  创建日期:2018/09/04
*  作者:Ellis
*  说明:暂无
*  修改日期:
*  作者:
*  说明:
*
*  修改日期:
*  作者:
*  说明:
******************************************************************/

//************************************头文件包含*******************************
#include <opencv2/opencv.hpp>
#include "opencv2/calib3d/calib3d.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include <stdio.h>
#include <vector>
#include <time.h>
//************************************************************************


//*************************************命名空间*********************************
using namespace cv;
using namespace std;
//****************************************************************************



//*************************************宏定义*******************************
#define wi 640
#define he 240
//*************************************************************************



//*************************************全局变量*******************************
clock_t time1, time2, time1_old, time2_old, time3;

Mat xyz;
Mat imgDisparity8U, imgDisparity16S;
Mat depth(imgDisparity8U.size(), CV_16UC1);
Mat pointClouds;
cv::Mat depthThresh;


bool down = false;
Point prept = Point(-1, -1);
Point curpt = Point(-1, -1);
//***************************************************************************


//******************************函数定义*************************************
void insertDepth32f(cv::Mat& depth);
void disp2Depth(cv::Mat dispMap, cv::Mat& depthMap, cv::Mat K);
void onMouse(int event, int x, int y, int flags, void * param);
void limit_depth(Mat& depth, Mat& l_depth);
void on_mouse(int event, int x, int y, int flags, void* ustc);
//***************************************************************************




/********************************************************************************************************
Function Name:main()
Author       :Ellis
Date         :2018/09/04
Description  :程序的入口函数
Parameter    :无
Notes        :无
Revision     :
*
*  修改日期:
*  作者:
*  说明:
********************************************************************************************************/
int main()
{
	//initialize and allocate memory to load the video stream from camera 
	VideoCapture camera(2);												//opencv的videocapture类，用于采集摄像头数据
	camera.set(CV_CAP_PROP_FRAME_WIDTH, wi);							//用set设置图像大小，视频传输会有很大的数据流量，如果图像太大，usb可能太慢
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, he);								
	if (!camera.isOpened())												//检测是否打开摄像头
		return 1;										

	Mat frame0, frame1, frame0_ud, frame1_ud, map1, map2, map3, map4;


	/************************************************************************/
	//左右相机内参，外参
	/************************************************************************/
	Mat cameraMatrix_Left = Mat::eye(3, 3, CV_64F);
	Mat distCoeffs_Left = Mat::zeros(5, 1, CV_64F);
	Mat cameraMatrix_Right = Mat::eye(3, 3, CV_64F);
	Mat distCoeffs_Right = Mat::zeros(5, 1, CV_64F);
	Mat R = Mat::zeros(3, 1, CV_64F);
	Mat T = Mat::zeros(3, 1, CV_64F);

	cameraMatrix_Left.at<double>(0, 0) = 408.74870;
	cameraMatrix_Left.at<double>(0, 2) = 157.13605;
	cameraMatrix_Left.at<double>(1, 1) = 408.90549;
	cameraMatrix_Left.at<double>(1, 2) = 119.50782;

	distCoeffs_Left.at<double>(0, 0) = -0.36491;
	distCoeffs_Left.at<double>(1, 0) = 0.00182;
	distCoeffs_Left.at<double>(2, 0) = 0.00331;
	distCoeffs_Left.at<double>(3, 0) = -0.00080;
	distCoeffs_Left.at<double>(4, 0) = 0;

	cameraMatrix_Right.at<double>(0, 0) = 400.19771;
	cameraMatrix_Right.at<double>(0, 2) = 147.69264;
	cameraMatrix_Right.at<double>(1, 1) = 400.46567;
	cameraMatrix_Right.at<double>(1, 2) = 101.55715;

	distCoeffs_Right.at<double>(0, 0) = -0.36531;
	distCoeffs_Right.at<double>(1, 0) = 0.08399;
	distCoeffs_Right.at<double>(2, 0) = 0.00362;
	distCoeffs_Right.at<double>(3, 0) = 0.00456;
	distCoeffs_Right.at<double>(4, 0) = 0;

	R.at<double>(0, 0) = -0.04235;
	R.at<double>(1, 0) = 0.00127;
	R.at<double>(2, 0) = -0.00054;

	T.at<double>(0, 0) = -167.03475;
	T.at<double>(1, 0) = -0.32197;
	T.at<double>(2, 0) = -3.69357;
	/************************************************************************/
	


	int ndisparities = 16;							/**< Range of disparity */
	int SADWindowSize = 9;							/**< Size of the block window. Must be odd */

	Mat frame;
	camera >> frame;

	Size imageSize;
	imageSize.height = frame.size().height;
	imageSize.width = frame.size().width / 2;

	Rect roi3, roi2;
	Mat Q, R1, P1, R2, P2;			//下一句是双目核心，输入量为左右相继内参，畸变矩阵，旋转平移矩阵，输出量为左右相机的r，p，q。。r，p用于生成畸变修复的map，q用于生成3D场景。roi2，3就是中间有画面的矩阵区域。
	stereoRectify(cameraMatrix_Left, distCoeffs_Left, cameraMatrix_Right, distCoeffs_Right, imageSize, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, imageSize, &roi2, &roi3);


	initUndistortRectifyMap(cameraMatrix_Left, distCoeffs_Left, R1, P1, imageSize, CV_16SC2, map1, map2);	//输入为r，p，得到修复畸变用的map

	initUndistortRectifyMap(cameraMatrix_Right, distCoeffs_Right, R2, P2, imageSize, CV_16SC2, map3, map4);	//修复畸变用的map



	while (true)
	{

		//----------------------计算帧率---------------------------//
		clock_t time2 = clock();
		////注意：除数不能为零，一定要进行非零判断
		if (time2 - time2_old == 0)
		{
			continue;
		}
		_cwprintf(L"fps%d\n", 1000 / (time2 - time2_old));
		time2_old = time2;
		//------------------------备注-----------------------------//
		//无
		//---------------------------------------------------------//



		Mat frame;
		camera >> frame;

		frame0 = frame(Rect(0, 0,frame.cols/2, frame.rows));
		frame1 = frame(Rect(wi/2, 0, frame.cols/2, frame.rows));

		remap(frame0, frame0_ud, map1, map2, INTER_LINEAR);			//畸变修复
		remap(frame1, frame1_ud, map3, map4, INTER_LINEAR);

		cvtColor(frame1_ud, frame1_ud, CV_BGR2GRAY);				//转为灰度，减少运算量
		cvtColor(frame0_ud, frame0_ud, CV_BGR2GRAY);

		Mat roi1 = frame1_ud;
		Mat roi0 = frame0_ud;

		//————————————————————————以下是SGBM算法————————————————————————————
		Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);

		int num_disp = (roi1.cols / 8 + 15) & -16;

		sgbm->setP1(8 *3*SADWindowSize*SADWindowSize);
		sgbm->setP2(32 *3*SADWindowSize*SADWindowSize);
		sgbm->setMinDisparity(0);
		sgbm->setNumDisparities(128);//128//num_disp
		sgbm->setUniquenessRatio(10);
		sgbm->setSpeckleWindowSize(100);
		sgbm->setSpeckleRange(32);
		sgbm->setDisp12MaxDiff(1);
		sgbm->setPreFilterCap(16);//32
		sgbm->setBlockSize(SADWindowSize);
		sgbm->setMode(StereoSGBM::MODE_SGBM);//StereoSGBM::MODE_HH
		sgbm->compute(roi0, roi1, imgDisparity16S);

		imgDisparity16S = imgDisparity16S.colRange(128, imgDisparity16S.cols);			//由于匹配后图像有一部分会无效，故剪掉
		imgDisparity16S.convertTo(imgDisparity8U, CV_8U, 255 / (128*16.));				//转为8U图，即可以显示的八位深度图

		imshow("Disparity", imgDisparity8U);

		cv::Mat disparityImage;
		if (disparityImage.empty() || disparityImage.type() != CV_8UC3 || disparityImage.size() != imgDisparity16S.size())
		{
			disparityImage = cv::Mat::zeros(imgDisparity16S.rows, imgDisparity16S.cols, CV_8UC3);
		}

		for (int x = 0; x < imgDisparity16S.rows; x++)
		{
			for (int y = 0; y < imgDisparity16S.cols; y++)
			{
				uchar val = imgDisparity8U.at<uchar>(x, y);
				uchar r, g, b;

				if (val == 0)
					r = g = b = 0;
				else
				{
					r = 255 - val;
					g = val < 128 ? val * 2 : (uchar)((255 - val) * 2);
					b = val;
				}

				disparityImage.at<cv::Vec3b>(x, y) = cv::Vec3b(r, g, b);

			}
		}

		imshow("disparityImage", disparityImage);

		reprojectImageTo3D(imgDisparity16S, pointClouds, Q, true);
		pointClouds *= 1.6;

		for (int x = 0; x < pointClouds.rows; ++x)
		{
			for (int y = 0; y < pointClouds.cols; ++y)
			{
				cv::Point3f point = pointClouds.at<cv::Point3f>(x, y);
				point.x = -point.x;
				pointClouds.at<cv::Point3f>(x, y) = point;
			}
		}

		imshow("disparitycolor", disparityImage);

		vector<Mat> xyzSet;
		split(pointClouds, xyzSet);
		Mat depth;
		xyzSet[2].copyTo(depth);

		double maxVal = 0, minVal = 0;
		depthThresh = Mat::zeros(depth.rows, depth.cols, CV_8UC1);
		minMaxLoc(depth, &minVal, &maxVal);
		double thrVal = minVal * 1.5;
		threshold(depth, depthThresh, thrVal, 255, CV_THRESH_BINARY_INV);
		depthThresh.convertTo(depthThresh, CV_8UC1);


		cvNamedWindow("depthThresh");
		//cvSetMouseCallback("depthThresh", on_mouse, 0);

		imshow("depthThresh", depthThresh);
		///////////////////////////////////////////////////////
		//xyz = xyz()
		//reprojectImageTo3D(imgDisparity16S, xyz, Q, false);									//用之前的Q将视差图转化为点云图
		//xyz = xyz * 16;			
		//imshow("xyz", xyz);
		//图片正常化	？？？

		//depth.create(imgDisparity8U.size(), CV_16UC1);
		//disp2Depth(imgDisparity16S, depth, cameraMatrix_Left);
		//imshow("depth0", depth);

		//insertDepth32f(depth);
		//imshow("depth", depth);

		//setMouseCallback("Disparity", onMouse, reinterpret_cast<void*> (&alineImgL));


		imshow("L", frame0);
		imshow("R", frame1);
		//imshow("L_Undistort", roi0);
		//imshow("R_Undistort", roi1);
		//Mat l_depth;
		//limit_depth(depth, l_depth);
		//imshow("l", l_depth);


		//Mat test_depth = depth;
		//Mat test_p = imgDisparity8U;

		waitKey(1);

	}

	return 0;
}



void disp2Depth(cv::Mat dispMap, cv::Mat &depthMap, cv::Mat K)
{
	int type = dispMap.type();

	double fx = K.at<double>(0, 0);
	double fy = K.at<double>(1, 1);
	double cx = K.at<double>(0, 2);
	double cy = K.at<double>(1, 2);
	float baseline = 167.03475; //基线距离65mm

	if (type == CV_16S)
	{
		const double PI = 3.14159265358;
		int height = dispMap.rows;
		int width = dispMap.cols;

		//uchar* dispData = (uchar*)dispMap.data;
		//ushort* depthData = (ushort*)depthMap.data;
		//float ttemp = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{

				if (!(dispMap.at<ushort>(i, j)))
				{
					depthMap.at<ushort>(i, j) = 60000;
					continue;  //防止0除

				}
					

				ushort s = ((ushort)(dispMap.at<ushort>(i, j)));
				ushort temp = (ushort)((float)fx *baseline / ((float)(dispMap.at<ushort>(i, j))));

				if (temp > 10000)
				{
					depthMap.at<ushort>(i, j) = 60000;
					continue;
				}
				depthMap.at<ushort>(i, j) = temp;
				//if (((float)fx *baseline / (temp)) > ttemp)
				//	ttemp = ((float)fx *baseline / (temp));
				//cout << (float)dispData[id] << endl;
			}
		}
		//cout << ttemp << endl;
	}
	else
	{
		cout << "please confirm dispImg's type!" << endl;
		cv::waitKey(0);
	}
}



void insertDepth32f(cv::Mat& depth)
{
	const int width = depth.cols;
	const int height = depth.rows;
	uchar* data = (uchar*)depth.data;
	cv::Mat integralMap = cv::Mat::zeros(height, width, CV_32F);
	cv::Mat ptsMap = cv::Mat::zeros(height, width, CV_32F);
	float* integral = (float*)integralMap.data;
	int* ptsIntegral = (int*)ptsMap.data;
	memset(integral, 0, sizeof(float) * width * height);
	memset(ptsIntegral, 0, sizeof(int) * width * height);
	for (int i = 0; i < height; ++i)
	{
		int id1 = i * width;
		for (int j = 0; j < width; ++j)
		{
			int id2 = id1 + j;
			if (data[id2] > 1e-3)
			{
				integral[id2] = data[id2];
				ptsIntegral[id2] = 1;
			}
		}
	}
	  // 积分区间
	for (int i = 0; i < height; ++i)
	{
		int id1 = i * width;
		for (int j = 1; j < width; ++j)
		{
			int id2 = id1 + j;
			integral[id2] += integral[id2 - 1];
			ptsIntegral[id2] += ptsIntegral[id2 - 1];
		}
	}
	for (int i = 1; i < height; ++i)
	{
		int id1 = i * width;
		for (int j = 0; j < width; ++j)
		{
			int id2 = id1 + j;
			integral[id2] += integral[id2 - width];
			ptsIntegral[id2] += ptsIntegral[id2 - width];
		}
	}
	int wnd;
	double dWnd = 2;
	while (dWnd > 1)
	{
		wnd = int(dWnd);
		dWnd /= 2;
		for (int i = 0; i < height; ++i)
		{
			int id1 = i * width;
			for (int j = 0; j < width; ++j)
			{
				int id2 = id1 + j;
				int left = j - wnd - 1;
				int right = j + wnd;
				int top = i - wnd - 1;
				int bot = i + wnd;
				left = max(0, left);
				right = min(right, width - 1);
				top = max(0, top);
				bot = min(bot, height - 1);
				int dx = right - left;
				int dy = (bot - top) * width;
				int idLeftTop = top * width + left;
				int idRightTop = idLeftTop + dx;
				int idLeftBot = idLeftTop + dy;
				int idRightBot = idLeftBot + dx;
				int ptsCnt = ptsIntegral[idRightBot] + ptsIntegral[idLeftTop] - (ptsIntegral[idLeftBot] + ptsIntegral[idRightTop]);
				double sumGray = integral[idRightBot] + integral[idLeftTop] - (integral[idLeftBot] + integral[idRightTop]);
				if (ptsCnt <= 0)
				{
					continue;
				}
				data[id2] = float(sumGray / ptsCnt);
			}
		}
		int s = wnd / 2 * 2 + 1;
		if (s > 201)
		{
			s = 201;
		}
		cv::GaussianBlur(depth, depth, cv::Size(s, s), s, s);
	}
}


void onMouse(int event, int x, int y, int flags, void * param)
{
	ushort valuess;
	Mat *im = reinterpret_cast<Mat*>(param);

	Mat test_imgDisparity8U = imgDisparity8U;
	Mat test_imgDisparity16S = imgDisparity16S;
	Mat test_depth = depth;


	switch (event)
	{
		case CV_EVENT_LBUTTONDOWN:     //鼠标左键按下响应：返回坐标和灰度 
		{
			if ((!(imgDisparity8U.at<uchar>(x, y)))
				||
				(!(imgDisparity16S.at<ushort>(x, y)))
				||
				(!(depth.at<ushort>(x, y)))
				)
			{
				break;
			}

			valuess = (ushort)(depth.at<ushort>(x, y));
			std::cout << "at(" << x << "," << y << ")value is: SGM: " << valuess << endl;

		}


		flags = false;
		break;
	}
}

void limit_depth(Mat& depth, Mat& l_depth)
{
	inRange(depth, 500,1500,l_depth);
}

void on_mouse(int event, int x, int y, int flags, void* ustc)
{
	int depth = depthThresh.at<short>(x, y);
	if (event == CV_EVENT_LBUTTONDOWN)    //右键按下
	{
		cout << "depth = " << depth << endl;
		prept = Point(x, y);
		down = true;
	}
	else if (event == CV_EVENT_LBUTTONUP)     //右键放开
		down = false;

}