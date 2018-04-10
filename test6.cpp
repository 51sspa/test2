// trainDetModifyMeanShape,����trainNew2Solve�޸�
// meanShape��Ϊams�й�һ�����ƽ����,��ƽ������ԭ��Ϊ���ģ����ź�����λ�ò���
#include "stdafx.h"
#include <iostream>  
#include <fstream>  
#include <string>  
#include <vector>  
#include <opencv2/opencv.hpp>
#include <random>

using namespace std;
using namespace cv;

void readPts(const string& absPtsName, Mat& landmarksMat, Point2f& faceCenterPt);
double gaussrand();
void myCalcDescriptors(Mat& image, Point2f keypoint, Mat& descriptor , bool isRotate);
void alignAPairOfShapesFunc(Mat shapeDst, Mat shapeSrc, Mat wMat, Mat& shapeRst, Mat& atMat = Mat());
void extractLandmarks(Mat srcLandmarkMat, Mat& dstLandmarkMat);
const int pointsNumTotal = 68;  // ��ǵ��ܵ���
const int pointsNumTrain = 49;  // һ��ѵ���ĵ���
const int perturbedNum = 10;    // ÿ��ͼ���Ŷ��Ĵ���
const int iterationNum = 6;
//const int startPoint = 17;  // ѵ����ʼ���ֵ
//const float faceSizeZoom = 200.0;   // ���ź��������С
//#define assignTranScl

int _tmain(int argc, _TCHAR* argv[])
{
	string trainFilePath = "E:\\FaceDatabase\\Mpie68Points\\trainsetCropZoom200Imsr\\";
	string trainFileList = "pngListPart.txt";

	ifstream trainFileFs(trainFilePath + trainFileList, ifstream::in);
	string trainFileName;
	Mat landmarksMat(2 * pointsNumTotal, 1, CV_32FC1);
	Point2f faceCenterPt(0.f, 0.f);
	// ������ѵ��ͼƬ��
	int trainFileNum = 0;
	while (getline(trainFileFs, trainFileName, '\n')) 
	{
		trainFileNum++;
	}
	
	// ��ȡ49�����meanShape
	Mat meanShape = Mat::zeros(2 * pointsNumTrain, 1,  CV_32FC1);
	FileStorage fsReadDetect("mpieHelenLfpw49PointsFlipFaceModel.yml", FileStorage::READ);
	fsReadDetect["MeanShape"] >> meanShape;
	fsReadDetect.release();

	/*
	//// ���Ի���ֵ��
	// ���Ժ󱣴�ͼƬ��·��
	const string savePath2 = "E:\\FaceDatabase\\lfpw\\trainsetMeanFace\\";
	// trainFileFs.close();
	// trainFileFs.open(trainFilePath + trainFileList, ifstream::in);
	trainFileFs.clear();
	trainFileFs.seekg(ios_base::beg);
	while (getline(trainFileFs, trainFileName, '\n')) 
	{
		cout << trainFilePath + trainFileName << endl;
		readPts(trainFilePath + trainFileName + ".pos", landmarksMat, faceCenterPt);
		// cout << faceCenterPt.x << faceCenterPt.y << endl;
		Mat tempImg = imread(trainFilePath + trainFileName + ".jpg", 0);
		// ����͵㿴Ч��
		//rectangle(tempImg, facePositionTest, facePositionTest2, Scalar(255,255,255), 2);
		for(int k = 0;k < pointsNumTotal;k++)
		{
			Point2f keyPointTest(meanValue.at<float>(2 * k, 0) + faceCenterPt.x, meanValue.at<float>(2 * k + 1, 0) + faceCenterPt.y);
			circle(tempImg, cv::Point((int)keyPointTest.x, (int)keyPointTest.y), 2, cv::Scalar(255,255,255), -1);
		}
		imwrite(savePath2 + trainFileName + ".jpg", tempImg);
	}
	*/
	int ii = 0;  // iiʼ�մ���ÿ��ѵ��ͼƬ
	#ifdef assignTranScl
		float scaleXStddevA = 0.1f;	
		float translationXStddevA = 5.f;	 
		float scaleYStddevA = 0.1f;	
		float translationYStddevA = 5.f;
		std::cout <<"scaleXMean:"<< 1.0 <<"    "<<"scaleXStddev:"<< scaleXStddevA <<endl;
		std::cout <<"translationXMean:"<< 0.0 <<"    "<<"translationXStddev:"<< translationXStddevA <<endl;
		std::cout <<"scaleYMean:"<< 1.0 <<"    "<<"scaleYStddev:"<< scaleYStddevA <<endl;
		std::cout <<"translationYMean:"<< 0.0 <<"    "<<"translationYStddev:"<< translationYStddevA <<endl;
	#else
		//// ���s,tx,ty�ľ�ֵ�ͱ�׼��
		Mat scaleMat(trainFileNum, 1, CV_32FC1);	// ÿ��ͼ�о�ֵ������ֵ��֮���x�����ϵ�����
		Mat translationXMat(trainFileNum, 1, CV_32FC1);	 // ÿ��ͼ�о�ֵ������ֵ��֮���x�����ϵ�λ��
		Mat translationYMat(trainFileNum, 1, CV_32FC1);
		trainFileFs.clear();
		trainFileFs.seekg(ios_base::beg);
		ii = 0;  // ii����ÿ��ѵ��ͼƬ
		while (getline(trainFileFs, trainFileName, '\n')) 
		{
			Mat landmarksTrainMat = Mat::zeros(2 * pointsNumTrain, 1, CV_32FC1);  // �Զ�ȡ�������б�ǵ�Ľض�
			readPts(trainFilePath + trainFileName + ".txt", landmarksMat, faceCenterPt);
			extractLandmarks(landmarksMat, landmarksTrainMat);
			// ����s,tx,ty
			Mat atMat = Mat::zeros(4, 1, CV_32FC1);
			alignAPairOfShapesFunc(landmarksTrainMat, meanShape, Mat(), Mat(), atMat);

			scaleMat.at<float>(ii, 0) = sqrtf(atMat.at<float>(0,0)*atMat.at<float>(0,0) + atMat.at<float>(1,0)*atMat.at<float>(1,0));
			translationXMat.at<float>(ii, 0) = atMat.at<float>(2,0);
			translationYMat.at<float>(ii, 0) = atMat.at<float>(3,0);;

			// ����translation��scale���ͼ���Ƿ���ȷ
			/*Mat debugTS(2 * pointsNumTrain, 1, CV_32FC1);
			Mat tempImg = imread(trainFilePath + trainFileName + ".png", 0);
			for(int k = 0;k < pointsNumTrain;k++)
			{
				debugTS.at<float>(2 * k, 0) = translationXMat.at<float>(ii, 0) + scaleMat.at<float>(ii, 0) * meanShape.at<float>(2 * k, 0);
				debugTS.at<float>(2 * k + 1, 0) = translationYMat.at<float>(ii, 0) + scaleMat.at<float>(ii, 0) * meanShape.at<float>(2 * k + 1, 0);
				circle(tempImg, cv::Point((int)(debugTS.at<float>(2 * k, 0)+ faceCenterPt.x), (int)(debugTS.at<float>(2 * k + 1, 0)+ faceCenterPt.y)), 2, cv::Scalar(255,255,255), -1);
			}
			imwrite("E:\\FaceDatabase\\Mpie68Points\\trainsetMeanTSFace\\" + trainFileName + ".png", tempImg);*/

			ii++;
		}

		// ��scale��translation�ľ�ֵ�ͱ�׼��
		Mat scaleMean, scaleStddev;	
		Mat translationXMean, translationXStddev;	 
		Mat translationYMean, translationYStddev;	 
		cv::meanStdDev(scaleMat, scaleMean, scaleStddev);
		cv::meanStdDev(translationXMat, translationXMean, translationXStddev);
		cv::meanStdDev(translationYMat, translationYMean, translationYStddev);
		std::cout <<"scaleMean:"<< scaleMean.at<double>(0,0)<<"    "<<"scaleStddev:"<<scaleStddev.at<double>(0,0)<<endl;
		std::cout <<"translationXMean:"<< translationXMean.at<double>(0,0)<<"    "<<"translationXStddev:"<<translationXStddev.at<double>(0,0)<<endl;
		std::cout <<"translationYMean:"<< translationYMean.at<double>(0,0)<<"    "<<"translationYStddev:"<<translationYStddev.at<double>(0,0)<<endl;
	#endif

	vector<Mat> preXVector;
	vector<Mat> RMergeVector;
	trainFileFs.clear();
	trainFileFs.seekg(ios_base::beg);
	// ��X����ֵ(�����200*200�����е�����)
	while (getline(trainFileFs, trainFileName, '\n')) 
	{
		for(int j = 0;j < perturbedNum;j++)
		{
			Mat preXMat(2 * pointsNumTrain, 1, CV_32FC1);
			float rand = (float)gaussrand();
			#ifdef assignTranScl
				float scaleX = (float)(1 + gaussrand() * scaleXStddevA);
				float translationX = (float)(gaussrand() * translationXStddevA);
				float scaleY = (float)(1 + gaussrand() * scaleYStddevA);
				float translationY = (float)(gaussrand() * translationYStddevA);
			#else
				float  scale = (float)(scaleMean.at<double>(0,0) + gaussrand() * scaleStddev.at<double>(0,0));
				float  translationX = (float)(translationXMean.at<double>(0,0) + gaussrand() * translationXStddev.at<double>(0,0));
				float  translationY = (float)(translationYMean.at<double>(0,0) + gaussrand() * translationYStddev.at<double>(0,0));
			#endif
			for(int k = 0;k < pointsNumTrain;k++)
			{
				preXMat.at<float>(2 * k, 0) = translationX + scale * meanShape.at<float>(2 * k, 0);
				preXMat.at<float>(2 * k + 1, 0) = translationY + scale * meanShape.at<float>(2 * k + 1, 0);
			}
			preXVector.push_back(preXMat);

			//// ���Բ鿴��ʼ��λ��
			//readPts(trainFilePath + trainFileName + ".txt", landmarksMat, faceCenterPt);
			//Mat tempImg = imread(trainFilePath + trainFileName + ".png", 0);
			//for (int k = 0 ; k < pointsNumTrain ; k++)
			//{
			//	circle(tempImg, cv::Point((int)(preXMat.at<float>(2 * k, 0) + faceCenterPt.x), (int)(preXMat.at<float>(2 * k + 1, 0) + faceCenterPt.y)), 2, cv::Scalar(255,255,255), -1);
			//}
			//char saveName[100];
			//sprintf(saveName, "E:\\FaceDatabase\\Mpie68Points\\initialPoints\\%s%d.png", trainFileName.c_str(), j);
			//imwrite(saveName,tempImg);
		}
	}

	// ��ʼѵ��(ѵ���۾���12����)
	for(int m = 0;m < iterationNum; m++)
	{
		std::cout << "iteration " << m << ".."<< endl;
		
		// �����RMerge
		Mat xDiffAllTran(trainFileNum * perturbedNum, 2 * pointsNumTrain, CV_32FC1);  // ����ͼ��xDiffת��
		Mat phiAllTran(trainFileNum * perturbedNum, 128 * pointsNumTrain + 1, CV_32FC1);        // ����ͼ���е��sift����ת��
		trainFileFs.clear();
		trainFileFs.seekg(ios_base::beg);
		ii = 0;
		while (getline(trainFileFs, trainFileName, '\n')) 
		{
			readPts(trainFilePath + trainFileName + ".txt", landmarksMat, faceCenterPt);
			Mat landmarksTrainMat = Mat::zeros(2 * pointsNumTrain, 1, CV_32FC1);  // �Զ�ȡ�������б�ǵ����ȡ
			extractLandmarks(landmarksMat, landmarksTrainMat);
			Mat tempImg = imread(trainFilePath + trainFileName + ".png", 0);
			for(int j = 0;j < perturbedNum;j++)
			{
				Mat phiSingle(1 , 128, CV_32FC1);  // ����ͼ�������sift����	
				// k��ֵ�����Ը���ѵ���������Ĳ�ͬ���ı�
				for(int k = 0;k < pointsNumTrain;k++)
				{
					xDiffAllTran.at<float>(ii, 2 * k) = landmarksTrainMat.at<float>(2 * k, 0) - preXVector[ii].at<float>(2 * k, 0);
					xDiffAllTran.at<float>(ii, 2 * k + 1) = landmarksTrainMat.at<float>(2 * k + 1, 0) - preXVector[ii].at<float>(2 * k + 1, 0);
					// ��֤ÿ����sift�����Ĺؼ��������ź��ͼ���в����
					float keyPointX = min(max((preXVector[ii].at<float>(2 * k, 0) + faceCenterPt.x), float(0)), float(tempImg.cols));
					float keyPointY = min(max((preXVector[ii].at<float>(2 * k + 1, 0) + faceCenterPt.y), float(0)), float(tempImg.rows));
					Point2f keyPoint(keyPointX, keyPointY);
					myCalcDescriptors(tempImg, keyPoint, phiSingle, false);
					for(int l = 0;l < 128;l++)
						phiAllTran.at<float>(ii, 128 * k + l) = phiSingle.at<float>(0, l);	
				}
				phiAllTran.at<float>(ii, 128 * pointsNumTrain) = 1.f;
				ii++;
			}
		}

		cout << norm(xDiffAllTran) <<endl;
		Mat RMergeTranMat(128 * pointsNumTrain + 1, 2 * pointsNumTrain, CV_32FC1);
		solve(phiAllTran, xDiffAllTran, RMergeTranMat, DECOMP_NORMAL | DECOMP_LU);
		Mat RMergeMat = RMergeTranMat.t();
		RMergeVector.push_back(RMergeMat);

		if(m == iterationNum - 1)
			break;

		// ����ÿһ��ͼpreX��λ��
		trainFileFs.clear();
		trainFileFs.seekg(ios_base::beg);
		ii = 0;
		while (getline(trainFileFs, trainFileName, '\n')) 
		{
			readPts(trainFilePath + trainFileName + ".txt", landmarksMat, faceCenterPt);
			Mat tempImg = imread(trainFilePath + trainFileName + ".png", 0);
			for(int j = 0;j < perturbedNum;j++)
			{
				Mat phiSingle(1 , 128, CV_32FC1);  // ����ͼ�������sift����
				Mat phiMerge(128 * pointsNumTrain + 1, 1, CV_32FC1);        // ����ͼ���е��sift��������һ��1
				for(int k = 0;k < pointsNumTrain;k++)
				{
					// ��֤ÿ����sift�����Ĺؼ��������ź��ͼ���в����
					float keyPointX = min(max((preXVector[ii].at<float>(2 * k, 0) + faceCenterPt.x), float(0)), float(tempImg.cols));
					float keyPointY = min(max((preXVector[ii].at<float>(2 * k + 1, 0) + faceCenterPt.y), float(0)), float(tempImg.rows));
					Point2f keyPoint(keyPointX, keyPointY);
					myCalcDescriptors(tempImg, keyPoint, phiSingle, false);
					for(int l = 0;l < 128;l++)
						phiMerge.at<float>(128 * k + l, 0) = phiSingle.at<float>(0, l);
				}
				phiMerge.at<float>(128 * pointsNumTrain, 0) = 1;
				// k��ֵ�����Ը���ѵ���������Ĳ�ͬ���ı�
				preXVector[ii] = preXVector[ii] + RMergeMat * phiMerge;
				ii++;
			}		
		}
		
	}	// ѵ��������

	// ����meanface,���ź�ƽ�Ƶľ�ֵ,RMerge
	FileStorage fsWrite("49PointsMpiePartModifyMeanShape_i6s200p10.yml", FileStorage::WRITE);
	fsWrite << "MeanShape" << meanShape;
	fsWrite << "MeanScale" << scaleMean.at<double>(0,0);
	fsWrite << "MeanTransX" << translationXMean.at<double>(0,0);
	fsWrite << "MeanTransY" << translationYMean.at<double>(0,0);
	for(int m = 0;m < iterationNum; m++)
	{
		char RName[10];
		sprintf(RName, "RMerge%d", m);
		fsWrite << RName << RMergeVector[m];
	}
			
}	// ��������


// �������ӱ�׼��̬�ֲ��������,��ָ������ΪE����׼��Ϊsd����ֻ������X = X * sd + E;
double gaussrand()
{
    static double V1, V2, S;
    static int phase = 0;
    double X;

    if ( phase == 0 ) {
        do {
            double U1 = (double)rand() / RAND_MAX;
            double U2 = (double)rand() / RAND_MAX;
             
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
         
        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
         
    phase = 1 - phase;
 
    return X;
}

//// ����:ʹ��solve������shapeSrc���뵽shapeDst, �����shapeRst
//// ����:shapeDst(2*n,1),shapeSrc(2*n,1),wMat(2*n,2*n)(Ȩ�ضԽǾ���,��wMatΪ�գ���ÿ��Ȩ�ض�һ��);
//// ���:shapeRst(2*n,1)(���Ժ�����shapeSrc��shapeDst��ͬ����ʱshapeSrc������),atMat(4,1):���ϵ��·ֱ���s*cosTheta,s*sinTheta,tx,ty
//// nΪѵ����ĸ���
void alignAPairOfShapesFunc(Mat shapeDst, Mat shapeSrc, Mat wMat, Mat& shapeRst, Mat& atMat)
{
	if(wMat.empty())
		wMat = Mat::eye(shapeDst.rows, shapeDst.rows, CV_32FC1);

	// �����������
	Mat src1 = Mat::zeros(shapeDst.rows, 4, CV_32FC1);
	for(int i = 0;i < shapeDst.rows;i += 2)
	{
		src1.at<float>(i, 0) = shapeSrc.at<float>(i, 0);
		src1.at<float>(i, 1) = -shapeSrc.at<float>(i + 1, 0);
		src1.at<float>(i, 2) = 1.f;
		src1.at<float>(i + 1, 0) = shapeSrc.at<float>(i + 1, 0);
		src1.at<float>(i + 1, 1) = shapeSrc.at<float>(i, 0);
		src1.at<float>(i + 1, 3) = 1.f;
	}
	//Mat atMat = Mat::zeros(4, 1, CV_32FC1);
	solve(src1, shapeDst, atMat, DECOMP_NORMAL | DECOMP_LU);
	if(shapeRst.empty())
		return;
	// ��shapeRst
	float tmp_x,tmp_y;
	for (int i = 0; i < shapeDst.rows;i += 2)
	{
		tmp_x = shapeSrc.at<float>(i,0);
		tmp_y = shapeSrc.at<float>(i + 1,0);

		shapeRst.at<float>(i,0) = atMat.at<float>(0,0) * tmp_x - atMat.at<float>(1,0) * tmp_y + atMat.at<float>(2,0);
		shapeRst.at<float>(i + 1,0) = atMat.at<float>(1,0) * tmp_x + atMat.at<float>(0,0) * tmp_y + atMat.at<float>(3,0);
	}
}

// ��ȡ��68����ǵ��е�49����
void extractLandmarks(Mat srcLandmarkMat, Mat& dstLandmarkMat)
{
	for(int k = 0;k < 51;k++)
	{
		if(k < 43)
		{
			dstLandmarkMat.at<float>(2 * k, 0) = srcLandmarkMat.at<float>((k + 17) * 2, 0);
			dstLandmarkMat.at<float>(2 * k + 1, 0) = srcLandmarkMat.at<float>((k + 17) * 2 + 1, 0);
		}
		else if(k == 43)
			continue;
		else if((43 < k) && (k < 47))
		{
			dstLandmarkMat.at<float>(2 * (k - 1), 0) = srcLandmarkMat.at<float>((k + 17) * 2, 0);
			dstLandmarkMat.at<float>(2 * (k - 1) + 1, 0) = srcLandmarkMat.at<float>((k + 17) * 2 + 1, 0);
		}
		else if(k == 47)
			continue;
		else if(k > 47)
		{
			dstLandmarkMat.at<float>(2 * (k - 2), 0) = srcLandmarkMat.at<float>(2 * (k + 17), 0);
			dstLandmarkMat.at<float>(2 * (k - 2) + 1, 0) = srcLandmarkMat.at<float>(2 * (k + 17) + 1, 0);
		}
	}
}