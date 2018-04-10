// testScore,测试训练出来的yml文件的误差程度
#include "stdafx.h"
#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>  
#include <opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;

void readPts(const string& absPtsName, Mat& landmarksMat, Point2f& faceCenterPt);
double gaussrand();
void myCalcDescriptors(Mat& image, Point2f keypoint, Mat& descriptor );
//double f(Mat& phi, Mat& w, double wb);
String face_cascade_name = "..\\Cascade\\haarcascade_frontalface_alt2.xml";
CascadeClassifier face_cascade;
const int pointsNumTotal = 68;  // 标记的总点数
const int pointsNumTrain = 24;  // 一共训练的点数
const int iterationNum = 5;
const int startPoint = 36;  // 训练起始点的值
const float faceSizeZoom = 200.0;

int _tmain(int argc, _TCHAR* argv[])
{
	// 读入检测yml文件
	FileStorage fsRead("68PointsEyeMouthMpiePart_i6s200p10ir1.0.yml", FileStorage::READ);
	Mat meanValue;   // 均值脸
	Vector<Mat> RVector;
	Vector<Mat> bVector;
	fsRead["MeanShape"] >> meanValue;
	for(int m = 0;m < iterationNum; m++)
	{
		char RName[5],bName[5];
		sprintf(RName, "R%d", m);
		sprintf(bName, "b%d", m);
		Mat RTemp,bTemp;
		fsRead[RName] >> RTemp;
		RVector.push_back(RTemp);
		fsRead[bName] >> bTemp;
		bVector.push_back(bTemp);
	}
	// 分别读入测试的3组yml
	Vector<Mat> wVector;
	Vector<float> wbVector;
	Mat wTemp;
	float wbTemp;
	FileStorage fsReadScore1("ScoreMpie1.yml", FileStorage::READ);
	fsReadScore1["w"] >> wTemp;
	wVector.push_back(wTemp);
	fsReadScore1["wb"] >> wbTemp;
	wbVector.push_back(wbTemp);
	FileStorage fsReadScore2("ScoreMpie2.yml", FileStorage::READ);
	fsReadScore2["w"] >> wTemp;
	wVector.push_back(wTemp);
	fsReadScore2["wb"] >> wbTemp;
	wbVector.push_back(wbTemp);
	FileStorage fsReadScore3("ScoreMpie3_is0.1it6.yml", FileStorage::READ);
	fsReadScore3["w"] >> wTemp;
	wVector.push_back(wTemp);
	fsReadScore3["wb"] >> wbTemp;
	wbVector.push_back(wbTemp);

	// 读入cascade
	if( !face_cascade.load( face_cascade_name ) )
		{ printf("--(!)Error loading\n"); return -1; };

	string testFilePath = "E:\\FaceDatabase\\Mpie68Points\\testsetCropZoom200\\";
	/*string testSavePath = "E:\\FaceDatabase\\Mpie68Points\\test\\";*/
	string testFileList = "pngList.txt";

	float errorSquareSumY[3] = {0};  // 不同yml的误差平方和
	float testFileNum = 0;  // 计算误差的图像总数
	ifstream testFileFs(testFilePath + testFileList, ifstream::in);
	string testFileName;
	Mat landmarksMat(2 * pointsNumTotal, 1, CV_32FC1);
	Point2f faceCenterPt(0.f, 0.f);
    while (getline(testFileFs, testFileName, '\n')) 
	{
			cout << testFileName <<endl;
			readPts(testFilePath + testFileName + ".txt", landmarksMat, faceCenterPt);
			Mat tempImg = imread(testFilePath + testFileName + ".png", 0);
			// 赋初始点(相对缩放人脸中心的坐标)
			Mat XTestMat(2 * pointsNumTrain, 1, CV_32FC1);
			for(int k = 0;k < pointsNumTrain;k++)
			{
				XTestMat.at<float>(2 * k, 0) = meanValue.at<float>(2 * (k + startPoint), 0);
				XTestMat.at<float>(2 * k + 1, 0) = meanValue.at<float>(2 * (k + startPoint) + 1, 0);
			}

			// 迭代
			for(int m = 0;m < iterationNum; m++)
			{
				Mat phiTest(128 * pointsNumTrain, 1, CV_32FC1);
				for(int k = 0;k < pointsNumTrain;k++)
				{
					// 保证每个求sift特征的关键点在缩放后的图像中不溢出
					float keyPointX = min(max((XTestMat.at<float>(2 * k, 0) + faceCenterPt.x), float(0)), float(tempImg.cols));
					float keyPointY = min(max((XTestMat.at<float>(2 * k + 1, 0) + faceCenterPt.y), float(0)), float(tempImg.rows));
					Point2f keyPointTest(keyPointX, keyPointY);
					Mat phiSingleTest(1 , 128, CV_32FC1);  // 单个点的sift特征
					myCalcDescriptors(tempImg, keyPointTest, phiSingleTest);
					for(int l = 0;l < 128;l++)
						phiTest.at<float>(128 * k + l, 0) = phiSingleTest.at<float>(0, l);
				}

				Mat R_mul_phi_Test(2 * pointsNumTrain, 1, CV_32FC1);
				gemm(RVector[m], phiTest, 1.0, Mat(), 0, R_mul_phi_Test, 0 );
				XTestMat = XTestMat + R_mul_phi_Test + bVector[m];
			}

			//// 计算真实y
			float errorSquareSum = 0;  // 每幅图每个点的误差平方和
			for(int k = 0;k < pointsNumTrain;k++)
			{
				// Point2f keyPointTest(XTestMat.at<float>(2 * k, 0) + faceCenterPt.x, XTestMat.at<float>(2 * k + 1, 0) + faceCenterPt.y);
				//circle(debugImage, cv::Point((int)XTrueZoomMat.at<float>(2 * k, 0), (int)XTrueZoomMat.at<float>(2 * k + 1, 0)), 2, cv::Scalar(0,0,0), -1);
				errorSquareSum += (landmarksMat.at<float>(2 * (k + startPoint), 0) - XTestMat.at<float>(2 * k, 0)) * (landmarksMat.at<float>(2 * (k + startPoint), 0) - XTestMat.at<float>(2 * k, 0));
				errorSquareSum += (landmarksMat.at<float>(2 * (k + startPoint) + 1, 0) - XTestMat.at<float>(2 * k + 1, 0)) * (landmarksMat.at<float>(2 * (k + startPoint) + 1, 0) - XTestMat.at<float>(2 * k + 1, 0));
			}
			float averageError = errorSquareSum / pointsNumTrain / 2;
			float singlePointError = std::sqrt(averageError) * 100 / faceSizeZoom;
			float yTrue = 2 * expf((-singlePointError + 2) / 2) - 2;

			//// 分别计算3组估算出来的RMSE
			// 计算phi
			Mat phi(128 * pointsNumTrain, 1, CV_32FC1);
			for(int k = 0;k < pointsNumTrain;k++)
			{
				// 保证每个求sift特征的关键点在缩放后的图像中不溢出
				float keyPointX = min(max((XTestMat.at<float>(2 * k, 0) + faceCenterPt.x), float(0)), float(tempImg.cols));
				float keyPointY = min(max((XTestMat.at<float>(2 * k + 1, 0) + faceCenterPt.y), float(0)), float(tempImg.rows));
				Point2f keyPointTest(keyPointX, keyPointY);
				Mat phiSingle(1 , 128, CV_32FC1);  // 单个点的sift特征
				myCalcDescriptors(tempImg, keyPointTest, phiSingle);
				for(int l = 0;l < 128;l++)
					phi.at<float>(128 * k + l, 0) = phiSingle.at<float>(0, l);
			}
			for(int g = 0;g < 3;g++)
			{
				Mat wTran;
				Mat wTran_mul_phi;
				transpose(wVector[g], wTran);
				gemm(wTran, phi, 1.0, Mat(), 0, wTran_mul_phi, 0 );
				float yEst = wTran_mul_phi.at<float>(0, 0) + wbVector[g];
				errorSquareSumY[g] += (yTrue - yEst) * (yTrue - yEst);
				cout << errorSquareSumY[g] << endl;
			}
			
			testFileNum++;
	}

	// 计算每次迭代的平均误差
	for(int g = 0;g < 3; g++)
	{
		float averageError = std::sqrt(errorSquareSumY[g] / testFileNum);
		cout <<  g + 1 << ":" << averageError << endl;
	}
}
