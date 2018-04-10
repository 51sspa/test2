// trainDetModifyMeanShape,基于trainNew2Solve修改
// meanShape改为ams中归一化后的平均脸,且平均脸以原点为重心，缩放后重心位置不变
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
const int pointsNumTotal = 68;  // 标记的总点数
const int pointsNumTrain = 49;  // 一共训练的点数
const int perturbedNum = 10;    // 每幅图像扰动的次数
const int iterationNum = 6;
//const int startPoint = 17;  // 训练起始点的值
//const float faceSizeZoom = 200.0;   // 缩放后的人脸大小
//#define assignTranScl

int _tmain(int argc, _TCHAR* argv[])
{
	string trainFilePath = "E:\\FaceDatabase\\Mpie68Points\\trainsetCropZoom200Imsr\\";
	string trainFileList = "pngListPart.txt";

	ifstream trainFileFs(trainFilePath + trainFileList, ifstream::in);
	string trainFileName;
	Mat landmarksMat(2 * pointsNumTotal, 1, CV_32FC1);
	Point2f faceCenterPt(0.f, 0.f);
	// 计算总训练图片数
	int trainFileNum = 0;
	while (getline(trainFileFs, trainFileName, '\n')) 
	{
		trainFileNum++;
	}
	
	// 读取49个点的meanShape
	Mat meanShape = Mat::zeros(2 * pointsNumTrain, 1,  CV_32FC1);
	FileStorage fsReadDetect("mpieHelenLfpw49PointsFlipFaceModel.yml", FileStorage::READ);
	fsReadDetect["MeanShape"] >> meanShape;
	fsReadDetect.release();

	/*
	//// 调试画均值脸
	// 测试后保存图片的路径
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
		// 画框和点看效果
		//rectangle(tempImg, facePositionTest, facePositionTest2, Scalar(255,255,255), 2);
		for(int k = 0;k < pointsNumTotal;k++)
		{
			Point2f keyPointTest(meanValue.at<float>(2 * k, 0) + faceCenterPt.x, meanValue.at<float>(2 * k + 1, 0) + faceCenterPt.y);
			circle(tempImg, cv::Point((int)keyPointTest.x, (int)keyPointTest.y), 2, cv::Scalar(255,255,255), -1);
		}
		imwrite(savePath2 + trainFileName + ".jpg", tempImg);
	}
	*/
	int ii = 0;  // ii始终代表每张训练图片
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
		//// 求解s,tx,ty的均值和标准差
		Mat scaleMat(trainFileNum, 1, CV_32FC1);	// 每幅图中均值脸与真值点之间的x方向上的缩放
		Mat translationXMat(trainFileNum, 1, CV_32FC1);	 // 每幅图中均值脸与真值点之间的x方向上的位移
		Mat translationYMat(trainFileNum, 1, CV_32FC1);
		trainFileFs.clear();
		trainFileFs.seekg(ios_base::beg);
		ii = 0;  // ii代表每张训练图片
		while (getline(trainFileFs, trainFileName, '\n')) 
		{
			Mat landmarksTrainMat = Mat::zeros(2 * pointsNumTrain, 1, CV_32FC1);  // 对读取出来所有标记点的截断
			readPts(trainFilePath + trainFileName + ".txt", landmarksMat, faceCenterPt);
			extractLandmarks(landmarksMat, landmarksTrainMat);
			// 计算s,tx,ty
			Mat atMat = Mat::zeros(4, 1, CV_32FC1);
			alignAPairOfShapesFunc(landmarksTrainMat, meanShape, Mat(), Mat(), atMat);

			scaleMat.at<float>(ii, 0) = sqrtf(atMat.at<float>(0,0)*atMat.at<float>(0,0) + atMat.at<float>(1,0)*atMat.at<float>(1,0));
			translationXMat.at<float>(ii, 0) = atMat.at<float>(2,0);
			translationYMat.at<float>(ii, 0) = atMat.at<float>(3,0);;

			// 调试translation和scale后的图像是否正确
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

		// 求scale和translation的均值和标准差
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
	// 给X赋初值(相对于200*200人脸中的坐标)
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

			//// 调试查看起始点位置
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

	// 开始训练(训练眼睛的12个点)
	for(int m = 0;m < iterationNum; m++)
	{
		std::cout << "iteration " << m << ".."<< endl;
		
		// 求矩阵RMerge
		Mat xDiffAllTran(trainFileNum * perturbedNum, 2 * pointsNumTrain, CV_32FC1);  // 所有图的xDiff转置
		Mat phiAllTran(trainFileNum * perturbedNum, 128 * pointsNumTrain + 1, CV_32FC1);        // 所有图所有点的sift特征转置
		trainFileFs.clear();
		trainFileFs.seekg(ios_base::beg);
		ii = 0;
		while (getline(trainFileFs, trainFileName, '\n')) 
		{
			readPts(trainFilePath + trainFileName + ".txt", landmarksMat, faceCenterPt);
			Mat landmarksTrainMat = Mat::zeros(2 * pointsNumTrain, 1, CV_32FC1);  // 对读取出来所有标记点的提取
			extractLandmarks(landmarksMat, landmarksTrainMat);
			Mat tempImg = imread(trainFilePath + trainFileName + ".png", 0);
			for(int j = 0;j < perturbedNum;j++)
			{
				Mat phiSingle(1 , 128, CV_32FC1);  // 单幅图单个点的sift特征	
				// k的值不可以根据训练用坐标点的不同而改变
				for(int k = 0;k < pointsNumTrain;k++)
				{
					xDiffAllTran.at<float>(ii, 2 * k) = landmarksTrainMat.at<float>(2 * k, 0) - preXVector[ii].at<float>(2 * k, 0);
					xDiffAllTran.at<float>(ii, 2 * k + 1) = landmarksTrainMat.at<float>(2 * k + 1, 0) - preXVector[ii].at<float>(2 * k + 1, 0);
					// 保证每个求sift特征的关键点在缩放后的图像中不溢出
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

		// 更新每一幅图preX的位置
		trainFileFs.clear();
		trainFileFs.seekg(ios_base::beg);
		ii = 0;
		while (getline(trainFileFs, trainFileName, '\n')) 
		{
			readPts(trainFilePath + trainFileName + ".txt", landmarksMat, faceCenterPt);
			Mat tempImg = imread(trainFilePath + trainFileName + ".png", 0);
			for(int j = 0;j < perturbedNum;j++)
			{
				Mat phiSingle(1 , 128, CV_32FC1);  // 单幅图单个点的sift特征
				Mat phiMerge(128 * pointsNumTrain + 1, 1, CV_32FC1);        // 单幅图所有点的sift特征最后多一个1
				for(int k = 0;k < pointsNumTrain;k++)
				{
					// 保证每个求sift特征的关键点在缩放后的图像中不溢出
					float keyPointX = min(max((preXVector[ii].at<float>(2 * k, 0) + faceCenterPt.x), float(0)), float(tempImg.cols));
					float keyPointY = min(max((preXVector[ii].at<float>(2 * k + 1, 0) + faceCenterPt.y), float(0)), float(tempImg.rows));
					Point2f keyPoint(keyPointX, keyPointY);
					myCalcDescriptors(tempImg, keyPoint, phiSingle, false);
					for(int l = 0;l < 128;l++)
						phiMerge.at<float>(128 * k + l, 0) = phiSingle.at<float>(0, l);
				}
				phiMerge.at<float>(128 * pointsNumTrain, 0) = 1;
				// k的值不可以根据训练用坐标点的不同而改变
				preXVector[ii] = preXVector[ii] + RMergeMat * phiMerge;
				ii++;
			}		
		}
		
	}	// 训练迭代完

	// 保存meanface,缩放和平移的均值,RMerge
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
			
}	// 主函数完


// 产生服从标准正态分布的随机数,若指定期望为E，标准差为sd，则只需增加X = X * sd + E;
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

//// 功能:使用solve函数将shapeSrc对齐到shapeDst, 并输出shapeRst
//// 输入:shapeDst(2*n,1),shapeSrc(2*n,1),wMat(2*n,2*n)(权重对角矩阵,若wMat为空，则每点权重都一样);
//// 输出:shapeRst(2*n,1)(可以和输入shapeSrc或shapeDst相同，此时shapeSrc被覆盖),atMat(4,1):从上到下分别是s*cosTheta,s*sinTheta,tx,ty
//// n为训练点的个数
void alignAPairOfShapesFunc(Mat shapeDst, Mat shapeSrc, Mat wMat, Mat& shapeRst, Mat& atMat)
{
	if(wMat.empty())
		wMat = Mat::eye(shapeDst.rows, shapeDst.rows, CV_32FC1);

	// 计算参数矩阵
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
	// 求shapeRst
	float tmp_x,tmp_y;
	for (int i = 0; i < shapeDst.rows;i += 2)
	{
		tmp_x = shapeSrc.at<float>(i,0);
		tmp_y = shapeSrc.at<float>(i + 1,0);

		shapeRst.at<float>(i,0) = atMat.at<float>(0,0) * tmp_x - atMat.at<float>(1,0) * tmp_y + atMat.at<float>(2,0);
		shapeRst.at<float>(i + 1,0) = atMat.at<float>(1,0) * tmp_x + atMat.at<float>(0,0) * tmp_y + atMat.at<float>(3,0);
	}
}

// 提取出68个标记点中的49个点
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