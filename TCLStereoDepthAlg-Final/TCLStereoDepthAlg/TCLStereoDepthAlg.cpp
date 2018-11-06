/*
 *  stereo_match.cpp
 *  calibration
 *
 *  
 *
 */

//#define saveImage 1

#include "OpenCVconfig.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/ximgproc/edge_filter.hpp"
#include "opencv2/ximgproc/disparity_filter.hpp"

#include <stdio.h>


using namespace cv;
using namespace cv::ximgproc;
using namespace std;

//static bool TCL_Mobile = 1;
//#define saveImage 1

static void print_help()
{
	//命令行参数：0037.png 0038.png --algorithm=sgbm --blocksize=5 --max-disparity=48 --scale=1.0 -o=0037-disp.bmp
	printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
	printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|sgbm3way] [--blocksize=<block_size>]\n"
		"[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
		"[--no-display] [-o=<disparity_image>] [-p=<point_cloud_file>]\n");
}

static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z = 1.0e4;
	FILE* fp = fopen(filename, "wt");
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}

int main(int argc, char** argv)
{
	
	//while (1) {
	std::string img1_filename = "";
	std::string img2_filename = "";

	std::string intrinsic_filename = "";
	std::string extrinsic_filename = "";
	std::string inverse_filename = "";//inverseMap.yml

	std::string disparity_filename = "";
	std::string point_cloud_filename = "";

	enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3, STEREO_3WAY = 4 };
	int alg = STEREO_SGBM;
	int SADWindowSize = 15;
	int numberOfDisparities = 32;
	bool no_display = 0;
	float scale = 1.0f;

	double wls_lambda = 8000.0;
	double wls_sigma = 1.5;

	bool TCL_Mobile = 1;

	Ptr<StereoBM> bm = StereoBM::create(16, 9);
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);
	
	//保存图像文件名后缀
	String suf = "_sgbm";

	int color_mode = alg == STEREO_BM ? 0 : -1;

	//参数命令行
	cv::CommandLineParser parser(argc, argv,
		"{@arg1|data//data//0099_aux.jpg|}{@arg2|data//data//0099_main.jpg|}{help h||}{algorithm|sgbm|}{max-disparity|32|}{blocksize|9|}{no-display||}{scale|1|}{i|intrinsics.yml|}{e|extrinsics.yml|}{vmap|inverseMap.yml|}{o||}{p||}{wls_lambda|8000.0|}{wls_sigma|1.5|}{TCL|1|}");
	if (parser.has("help"))
	{
		print_help();
		return 0;
	}
	img1_filename = parser.get<std::string>(0);
	img2_filename = parser.get<std::string>(1);
	if (parser.has("algorithm"))
	{
		std::string _alg = parser.get<std::string>("algorithm");
		alg = _alg == "bm" ? STEREO_BM :
			_alg == "sgbm" ? STEREO_SGBM :
			_alg == "hh" ? STEREO_HH :
			_alg == "var" ? STEREO_VAR :
			_alg == "sgbm3way" ? STEREO_3WAY : -1;
	}
	numberOfDisparities = parser.get<int>("max-disparity");
	SADWindowSize = parser.get<int>("blocksize");
	scale = parser.get<float>("scale");
	TCL_Mobile = parser.get<bool>("TCL");
	no_display = parser.has("no-display");
	if (parser.has("i"))
		intrinsic_filename = parser.get<std::string>("i");
	if (parser.has("e"))
		extrinsic_filename = parser.get<std::string>("e");
	if (parser.has("vmap"))
		inverse_filename = parser.get<std::string>("vmap");
	if (parser.has("o"))
		disparity_filename = parser.get<std::string>("o");
	if (parser.has("p"))
		point_cloud_filename = parser.get<std::string>("p");
	wls_lambda = parser.get<double>("wls_lambda");
	wls_sigma = parser.get<double>("wls_sigma");
	if (!parser.check())
	{
		parser.printErrors();
		return 1;
	}

	

	//Mat img1 = imread(img1_filename, color_mode);
	//Mat img2 = imread(img2_filename, color_mode);
	Mat img1 = imread(img1_filename, 1);
	Mat img2 = imread(img2_filename, 1);

	
	//Mat color_boost1, color_boost2;
	//decolor(img1, img1, color_boost1);
	//decolor(img2, img2, color_boost2);
	
	if (img1.empty())
	{
		printf("Command-line parameter error: could not load the first input image file\n");
		return -1;
	}
	if (img2.empty())
	{
		printf("Command-line parameter error: could not load the second input image file\n");
		return -1;
	}

	if (scale != 1.f)
	{
		Mat temp1, temp2;
		int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
		resize(img1, temp1, Size(), scale, scale, method);
		img1 = temp1;
		resize(img2, temp2, Size(), scale, scale, method);
		img2 = temp2;
	}
	

	Size img_size = img1.size();

	Rect roi1, roi2;
	Mat Q;
	
	Size cropSize;
	
	if (TCL_Mobile)
	{
		
		// reading intrinsic parameters
		FileStorage fs(intrinsic_filename, FileStorage::READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", intrinsic_filename.c_str());
			return -1;
		}

		Mat M1, D1, M2, D2;
		fs["M1"] >> M1;
		fs["D1"] >> D1;
		fs["M2"] >> M2;
		fs["D2"] >> D2;

		M1 *= scale;
		M2 *= scale;

		fs.open(extrinsic_filename, FileStorage::READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", extrinsic_filename.c_str());
			return -1;
		}

		Mat R, T, R1, P1, R2, P2;
		Rect validROI;//有效的公共区域
		fs["R"] >> R;
		fs["T"] >> T;
		fs["R1"] >> R1;
		fs["P1"] >> P1;
		fs["R2"] >> R2;
		fs["P2"] >> P2;
		fs["roi_rec"] >> validROI;
		P2.at<double>(0, 2) -= 2;

		
		
		//stereoRectify(M1, D1, M2, D2, img_size,R, T, R1, R2, P1, P2, Q, 0, 1, img_size, &roi1, &roi2);
		
		Mat map11, map12, map21, map22;
		initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
		initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

		Mat img1r, img2r;
		remap(img1, img1r, map11, map12, INTER_LINEAR);
		remap(img2, img2r, map21, map22, INTER_LINEAR);

		Mat img1Flip, img2Flip, img1f, img2f;
		img1Flip = img1r(validROI).clone();
		img2Flip = img2r(validROI).clone();
		
		flip(img1Flip, img1f, -1);
		flip(img2Flip, img2f, -1);

		img1 = img2f;
		img2 = img1f;

		cropSize = img1.size();

		resize(img1, img1, Size(1024, 768));
		resize(img2, img2, Size(1024, 768));

		String rec_name1 = img1_filename.substr(0, img1_filename.length() - 4) + "_rec.jpg";
		String rec_name2 = img2_filename.substr(0, img2_filename.length() - 4) + "_rec.jpg";

		//imwrite(rec_name1, img2);
		//imwrite(rec_name2, img1);
	}
	
	
	
	GaussianBlur(img1, img1, Size(3, 3), 2, 2);
	
	
	
	numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width / 8) + 15) & -16;

	bm->setROI1(roi1);
	bm->setROI2(roi2);
	bm->setPreFilterCap(31);
	bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
	bm->setMinDisparity(0);
	bm->setNumDisparities(numberOfDisparities);
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(15);
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(1);

	sgbm->setPreFilterCap(63);
	int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
	sgbm->setBlockSize(sgbmWinSize);

	int cn = img1.channels();

	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(numberOfDisparities);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleWindowSize(800);
	sgbm->setSpeckleRange(4);
	sgbm->setDisp12MaxDiff(0);
	if (alg == STEREO_HH)
		sgbm->setMode(StereoSGBM::MODE_HH);
	else if (alg == STEREO_SGBM)
		sgbm->setMode(StereoSGBM::MODE_SGBM);
	else if (alg == STEREO_3WAY)
		sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);
	
	Mat disp, disp8;
	
	bool makeBorder = 1;
	if (makeBorder)
	{
		//解决视差黑边
		Mat img1p, img2p, dispp;
		copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
		copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

		//while (1)
		//{
		int64 t = getTickCount();
		if (alg == STEREO_BM)
			bm->compute(img1p, img2p, dispp);
		else if (alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY) 
			sgbm->compute(img1p, img2p, dispp);		
		t = getTickCount() - t;
		printf("Time elapsed: %fms\n", t * 1000 / getTickFrequency());
		//}
		//解决视差黑边
		disp = dispp.colRange(numberOfDisparities, img1p.cols);
	}
	else
	{

		int64 t = getTickCount();
		if (alg == STEREO_BM)
			bm->compute(img1, img2, disp);
		else if (alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY)
			sgbm->compute(img1, img2, disp);
			

		t = getTickCount() - t;
		printf("SGBM Time elapsed: %fms\n", t * 1000 / getTickFrequency());
	}

	
	if (alg != STEREO_VAR)
		disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
	else
		disp.convertTo(disp8, CV_8U);
	


	///*** 导向滤波 guidedFilter ***///
	//Mat guidedDisp;
	//float eps = 0.001 * 255 * 255;//eps的取值很关键（乘以 255的平方） 
	////Mat dst;
	////Mat mask = imread("test3-gt-720P.bmp", 0);
	//double filtering_time0 = (double)getTickCount();
	//cv::ximgproc::guidedFilter(img1, disp8, guidedDisp,5, eps, -1);
	//filtering_time0 = ((double)getTickCount() - filtering_time0) / getTickFrequency();
	//cout << "GUI filter Time Elapsed: " << filtering_time0 * 1000.0 << endl;

	
//#ifdef saveImage
	//String nameGuided = img1_filename.substr(0, img1_filename.length() - 4) + suf + "_guidedFilter.bmp";
	//imwrite(nameGuided, guidedDisp);
//#endif

	///****WLS filter:disparity post-process***///
	Mat filterDisp;
	disp.convertTo(filterDisp, CV_16S);
	
	Mat wlsDisp;
	Ptr<DisparityWLSFilter> wls_filter;

	wls_filter = createDisparityWLSFilterGeneric(false);
	wls_filter->setDepthDiscontinuityRadius((int)ceil(0.5*sgbmWinSize));

	
	//double lambda = 8000.0;
	//double sigma = 1.5;
	double lambda = wls_lambda;
	double sigma = wls_sigma;

	//! [filtering]
	wls_filter->setLambda(lambda);
	wls_filter->setSigmaColor(sigma);
	double filtering_time = (double)getTickCount();
	wls_filter->filter(filterDisp, img1, wlsDisp);
	filtering_time = ((double)getTickCount() - filtering_time) / getTickFrequency();
	cout <<"WLS filter Time Elapsed: "<< filtering_time * 1000.0 << endl;
	//! [filtering]
	//Mat conf_map = wls_filter->getConfidenceMap();

	//// Get the ROI that was used in the last filter call:
	//Rect ROI = wls_filter->getROI();

	
	wlsDisp.convertTo(wlsDisp, CV_8U, 255 / (numberOfDisparities*16.));
	

#ifdef saveImage
	//String suf = "_bm_720P.bmp";
	String nameWLS = img2_filename.substr(0, img2_filename.length() - 4) + suf + "_wlsFilter.bmp";
	imwrite(nameWLS, wlsDisp);
#endif


	int64 t2;// = getTickCount();
	//视差图映射回主摄像头图像
	if (TCL_Mobile)
	{
		resize(img1, img1, cropSize);
		resize(img2, img2, cropSize);
		resize(wlsDisp, wlsDisp, cropSize);

		t2 = getTickCount();

		FileStorage fsInverseMap(inverse_filename, FileStorage::READ);
		Mat MapX2, MapY2, simg, rimg;
		fsInverseMap["MapX2"] >> MapX2;
		fsInverseMap["MapY2"] >> MapY2;

		MapX2.convertTo(MapX2, CV_32FC1);
		MapY2.convertTo(MapY2, CV_32FC1);

		fsInverseMap.open(extrinsic_filename, FileStorage::READ);
		Rect validROI;
		fsInverseMap["roi_rec"] >> validROI;

		t2 = getTickCount() - t2;
		printf("ALL Time elapsed: %fms\n", t2 * 1000 / getTickFrequency());

		simg = Mat::zeros(img_size, CV_8U);

		flip(wlsDisp, wlsDisp, -1);
		wlsDisp.copyTo(simg(validROI));

		remap(simg, rimg, MapX2, MapY2, INTER_LINEAR);

		disp8 = rimg;
		//imshow("rimg", rimg);
		//waitKey();
	}
	else
	{
		disp8 = wlsDisp;
	}
	
	
	
	
	/*ofstream fs("map.xls");
	fs << format(disp8, Formatter::FMT_CSV);
	fs.close();
*/
	
	//伪彩色显示
	Mat dispPsuedo;
#ifdef saveImage
	String srcDisp = img2_filename.substr(0, img2_filename.length() - 4) + suf + ".bmp"; //原始视差图
	String nameP = img1_filename.substr(0, img1_filename.length() - 4) + suf + "_psuedo.bmp";//原始视差图伪彩图
	//String namePguided = img1_filename.substr(0, img1_filename.length() - 4) + suf + "_guidedFilter_psuedo.bmp";//guided filter伪彩图
	String namePwls = img1_filename.substr(0, img1_filename.length() - 4) + suf + "_wlsFilter_psuedo.bmp";//wls filter伪彩图
	
	imwrite(srcDisp, disp8);

	//applyColorMap(disp8, dispPsuedo, COLORMAP_JET);
	//imwrite(nameP, dispPsuedo);

	//applyColorMap(guidedDisp, dispPsuedo, COLORMAP_JET);
	//imwrite(namePguided, dispPsuedo);
	
	//applyColorMap(wlsDisp, dispPsuedo, COLORMAP_JET);
	//imwrite(namePwls, dispPsuedo);

#endif


	if (!no_display)
	{
		namedWindow("left", WINDOW_KEEPRATIO);
		imshow("left", img1);
		namedWindow("right", WINDOW_KEEPRATIO);
		imshow("right", img2);
		namedWindow("disparity", WINDOW_KEEPRATIO);
		imshow("disparity", disp8);
		printf("press any key to continue...");
		fflush(stdout);
		waitKey(0);
		printf("\n");
	}

	
	
	if (!disparity_filename.empty())
		imwrite(disparity_filename, disp8);

	if (!point_cloud_filename.empty())
	{
		printf("storing the point cloud...");
		fflush(stdout);
		Mat xyz;
		reprojectImageTo3D(disp, xyz, Q, true);
		saveXYZ(point_cloud_filename.c_str(), xyz);
		printf("\n");
	}

	

	return 0;
}