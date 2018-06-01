#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include "precomp2.hpp"
#include "edge_detection.hpp"

using namespace std;
using namespace cv;


static inline int SSD(Vec3b p1, Vec3b p2) {
	int x0 = p1[0], y0 = p1[1], z0 = p1[2];
	int x1 = p2[0], y1 = p2[1], z1 = p2[2];
	int x = x1 - x0, y = y1 - y0, z = z1 - z0;
	return x*x+y*y+z*z;
}
struct myDMatch {
	int queryIdx; // query descriptor index
	int trainIdx_first; // train descriptor index
	int trainIdx_second; // train descriptor index
	float distance_first;
	float distance_second;
	//myDMatch() {};
};
float calDistance(float*r1, float*r2, int len) {
	float result = 0.f;
	for (int i = 0; i < len; i++) {
		result += (r1[i] - r2[i])*(r1[i] - r2[i]);
	}
	result = sqrtf(result);
	return result;
}
void my_getTop2(Mat&descriptor1, Mat&descriptor2, vector<myDMatch>&matches) {
	matches.resize(descriptor1.rows);
	for (int i = 0; i < descriptor1.rows; i++) {
		myDMatch match;
		match.distance_first = DBL_MAX; match.distance_second = DBL_MAX;
		match.queryIdx = i;
		match.trainIdx_first = -1; match.trainIdx_second = -1;
		float*descriptor1_row = descriptor1.ptr<float>(i);
		for (int j = 0; j < descriptor2.rows; j++) {
			float*descriptor2_row = descriptor2.ptr<float>(j);
			float dis = calDistance(descriptor1_row, descriptor2_row, descriptor1.cols);
			if (dis < match.distance_first) {
				match.distance_second = match.distance_first;
				match.distance_first = dis;
				match.trainIdx_second = match.trainIdx_first;
				match.trainIdx_first = j;
			}
			else if (dis < match.distance_second) {
				match.distance_second = dis;
				match.trainIdx_second = j;
			}
		}
		matches.push_back(match);
	}
}
void my_filterRatio(vector<myDMatch>&src, vector<DMatch>&dst, float th_ratio) {
	dst.clear();
	for (int i = 0; i < src.size(); i++) {
		myDMatch temp = src[i];
		float ratio = temp.distance_first / temp.distance_second;
		if (ratio < th_ratio) {
			DMatch match(temp.queryIdx, temp.trainIdx_first, temp.distance_first);
			dst.push_back(match);
		}
	}
}

void findMatchPoints(Mat&imgL, Mat&imgR, vector<KeyPoint>&kp1, vector<KeyPoint>&kp2,
	Mat&des1, Mat&des2,
	vector<Point2f> &points_l, vector<Point2f>& points_r,
	vector<DMatch>&matches, Mat&F) {
	Mat mask;
	int nfeature = 100;
	SIFT sift1(nfeature), sift2(nfeature);

	sift1(imgL, mask, kp1, des1);
	sift2(imgR, mask, kp2, des2);

	vector<myDMatch>mymatches;
	vector<DMatch>ratio_matches;
	//Ptr<cv::DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	//matcher->match(des1, des2, ratio_matches);
	my_getTop2(des1, des2, mymatches);
	my_filterRatio(mymatches, ratio_matches, 0.8);

	//vector<Point2f> points_l(ratio_matches.size()), points_r(ratio_matches.size());
	points_l.resize(ratio_matches.size());
	points_r.resize(ratio_matches.size());
	for (int i = 0; i < ratio_matches.size(); i++) {
		points_l[i] = kp1[ratio_matches[i].queryIdx].pt;
		points_r[i] = kp2[ratio_matches[i].trainIdx].pt;
	}

	vector<uchar>inlierMask(ratio_matches.size());
	//F = findHomography(points_l, points_r, inlierMask, CV_RANSAC);
	F = findFundamentalMat(points_l, points_r, inlierMask, CV_RANSAC);

	for (int i = 0; i < inlierMask.size(); i++) {
		if (inlierMask[i]) {
			matches.push_back(ratio_matches[i]);
		}
	}
}

void getRectifyImg(Mat&src, Mat&dst, Mat&H,
	vector<Point2f> &points_l, vector<Point2f>& points_r
	) {

}

void drawEpipolarLines(Mat& imgL, Mat& imgR,
	vector<Point2f> &points_l, vector<Point2f>& points_r,
	vector<Vec3f>&lines1, vector<Vec3f>&lines2, 
	Mat&F,Mat&out) {
	//dst_imgL.create(imgL.size(), imgL.type());
	//dst_imgR.create(imgR.size(), imgR.type());
	Mat outImg(imgL.rows, imgL.cols * 2, CV_8UC3);
	Rect rect1(0, 0, imgL.cols, imgL.rows);
	Rect rect2(imgL.cols, 0, imgL.cols, imgL.rows);
	imgL.copyTo(outImg(rect1));
	imgR.copyTo(outImg(rect2));

	//get lines
	computeCorrespondEpilines(points_l, 1, F, lines1);
	computeCorrespondEpilines(points_r, 2, F, lines2);

	RNG rng(0);
	for (int i = 0; i < points_l.size(); i++) {
		Scalar color(rng(256), rng(256), rng(256));
		line(outImg(rect2),
			Point(0, -lines1[i][2] / lines1[i][1]),
			Point(imgL.cols, -(lines1[i][2] + lines1[i][0] * imgL.cols) / lines1[i][1]),
			color);
		circle(outImg(rect1), points_l[i], 3, color);
		
		line(outImg(rect1),
			Point(0, -lines2[i][2] / lines2[i][1]),
			Point(imgR.cols, -(lines2[i][2] + lines2[i][0] * imgR.cols) / lines2[i][1]),
			color);
		circle(outImg(rect2), points_r[i], 3, color);
	}

	out = outImg;
}
void LoG(const cv::Mat& src, cv::Mat& dst, float threshold=0.4) {
	Mat gau31 = my_getGaussianKernel(gaussian_ksize, -1);
	Mat gau_kern = gau31*gau31.t();

	Mat gau_x, gau_y;
	my_getDerivative(gau_kern, gau_x, 1, 1);
	gau_y = gau_x.t();
	Mat src_x, src_y;
	filter2D(src, dst, src.depth(), gau_x);
	//filter2D(src, src_y, src.depth(), gau_y);
	//my_magnitude(src_x, src_y, dst);
	//normalize(dst, dst, 1, 0, NORM_MINMAX);
	//normalize(dst, dst, 1, 0, NORM_MINMAX);

	//GaussianBlur(src, src, Size(3,3), 0);
	//Mat sobel=(Mat_<float>(3, 3) << -1, 0, 1,
	//	-2, 0, 2,
	//	-1, 0, 1);
	//filter2D(src, dst, src.depth(), sobel);
	//normalize(dst, dst, 1, 0, NORM_MINMAX);
	//for (int y = 0; y < dst.rows; y++) {
	//	float*row = dst.ptr<float>(y);
	//	for (int x = 0; x < dst.cols; x++) {
	//		if (row[x] < -threshold)
	//			row[x] = 2 * threshold;
	//		else if (row[x] > threshold)
	//			row[x] = 2 * threshold;
	//		else
	//			row[x] += threshold;
	//	}
	//}
}
void getDisparityMap(Mat&imgL,Mat&imgR,Mat&dst,int winsize=5,int maxOffset=30,int minOffset=0) {
	Mat _imgL, _imgR;
	_imgL = imgL;
	_imgR = imgR;
	//normalize(imgL, _imgL, 1, 0, NORM_MINMAX);
	//normalize(imgR, _imgR, 1, 0, NORM_MINMAX);
	LoG(imgL, _imgL);
	LoG(imgR, _imgR);

	//Mat window(winsize, winsize, CV_32F);
	Mat out(imgL.size(), CV_32F);
	int half_winsize = winsize / 2;
	int r = imgL.rows, c = imgL.cols;

	for (int y = half_winsize; y < r - half_winsize; y++) {
		//Vec3b* imgL_row = imgL.ptr<Vec3b>(y);
		//Vec3b* imgR_row = imgR.ptr<Vec3b>(y);
		for (int x = half_winsize; x < c - half_winsize; x++) {
			float optimal_offset = 0, optimal_offset_r = 0;
			float optimal_score = FLT_MAX, optimal_score_r = FLT_MAX;
			if (x == 71 && y == 227)
				cout << "here";

			for (int offset = minOffset; offset < maxOffset; offset++) {
				float score = 0.f;
				if (offset + x + half_winsize >= c)
				//if (x + half_winsize - offset < 0)
					continue;

				for (int yy = -half_winsize; yy < half_winsize; yy++) {
					float*row1 = _imgL.ptr<float>(y + yy);
					float*row2 = _imgR.ptr<float>(y + yy);
					for (int xx = -half_winsize; xx < half_winsize; xx++) {
						float p1 = row1[x + xx];
						float p2 = row2[x + xx + offset];
						score += (p1 - p2)*(p1 - p2);
					}
				}
				if (optimal_score > score) {
					optimal_score = score;
					optimal_offset = offset;
				}
			}
			//bidirection matching
			int x_r=x-optimal_offset;
			//if (x_r + half_winsize >= c) {
			//	//optimal_offset = 0;
			//}
			if (x_r - half_winsize < 0)
				x_r = half_winsize;
			for (int offset_r = minOffset; offset_r < maxOffset; offset_r++) {
				if (x_r + offset_r + half_winsize >= c)
					break;
				float score_r = 0.f;
				for (int yy_r = -half_winsize; yy_r < half_winsize; yy_r++) {
					float*row1 = _imgL.ptr<float>(y + yy_r);
					float*row2 = _imgR.ptr<float>(y + yy_r);
					for (int xx_r = -half_winsize; xx_r < half_winsize; xx_r++) {
						float p1 = row1[x + xx_r + offset_r];
						float p2 = row2[x + xx_r];
						score_r += (p1 - p2)*(p1 - p2);
					}
				}
				if (optimal_score_r > score_r) {
					optimal_score_r = score_r;
					optimal_offset_r = offset_r;
				}
			}
			//if (abs(optimal_offset_r - optimal_offset)>maxOffset/8)
			//if(abs(optimal_score-optimal_score_r)>min(optimal_score, optimal_score_r))
			optimal_offset = optimal_offset_r;
			
			out.at<float>(y, x) = (optimal_offset)/(maxOffset);
		}
	}
	normalize(out, dst, 1, 0, NORM_MINMAX);
	//dst = out;
}
int main() {
	//Mat imgL = imread("../data/view1.png",CV_LOAD_IMAGE_GRAYSCALE);
	//Mat imgR = imread("../data/view0.png",CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgL = imread("../data/im2.png",CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgR = imread("../data/im6.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat imgL = imread("../data/im0.png",CV_LOAD_IMAGE_GRAYSCALE);
	//Mat imgR = imread("../data/im1.png",CV_LOAD_IMAGE_GRAYSCALE);
	//Mat imgL = imread("../data/Left3.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	//Mat imgR = imread("../data/Right3.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	//Mat imgL = imread("../data/NotreDame1.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	//Mat imgR = imread("../data/NotreDame2.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	imgL.convertTo(imgL, CV_32F);
	imgR.convertTo(imgR, CV_32F);
	//normalize(imgL, imgL, 1, 0, NORM_MINMAX);
	//normalize(imgR, imgR, 1, 0, NORM_MINMAX);

	int maxOffset = (imgL.cols / 8 + 15)&-16;
	Mat unrectified_map;
	getDisparityMap(imgL, imgR, unrectified_map,7,maxOffset);
	normalize(unrectified_map, unrectified_map, 255, 0, NORM_MINMAX);
	unrectified_map.convertTo(unrectified_map, CV_8U, 1, 0);
	imshow("disparity_unretified", unrectified_map);
	filterSpeckles(unrectified_map, 150, 100, 30);
	imshow("disparity_unretified_pack", unrectified_map);
	imwrite("../data/disparity_unretified.png", unrectified_map);

	vector<KeyPoint>keypoints1, keypoints2;
	Mat descriptor1, descriptor2;
	vector<DMatch>matches;
	vector<Point2f> points_l, points_r;
	Mat F;
	Mat match;
	imgL = imread("../data/im2.png");
	imgR = imread("../data/im6.png");
	findMatchPoints(imgL, imgR, keypoints1, keypoints2, descriptor1, descriptor2, points_l, points_r, matches, F);
	drawMatches(imgL, keypoints1, imgR, keypoints2, matches, match, Scalar::all(-1));
	imwrite("../data/match.jpg", match);

	//get lines
	vector<Vec3f>lines1; vector<Vec3f>lines2;
	Mat line_out;
	drawEpipolarLines(imgL, imgR, points_l, points_r,lines1,lines2, F,line_out);
	imshow("epipolar_lines", line_out);
	imwrite("../data/epipolar_lines.png", line_out);
	//get rectified images
	Mat H1, H2;
	stereoRectifyUncalibrated(points_l, points_r, F, imgL.size(), H1, H2);
	cout << H1<<endl; cout << H2<<endl;
	Mat rectified1, rectified2;
	warpPerspective(imgL, rectified1, H1, imgL.size());
	imshow("imgL", rectified1);
	imwrite("../data/r_L.png", rectified1);
	warpPerspective(imgR, rectified2, H2, imgL.size());
	imshow("imgR", rectified2);
	imwrite("../data/r_R.png", rectified2);

	//Mat rectified_map;
	//cvtColor(rectified1, rectified1, CV_RGB2GRAY);
	//cvtColor(rectified2, rectified2, CV_RGB2GRAY);
	//rectified1.convertTo(rectified1, CV_32F);
	//rectified2.convertTo(rectified2, CV_32F);
	//getDisparityMap(rectified1, rectified2, rectified_map, 7, maxOffset-10);
	//normalize(rectified_map, rectified_map, 255, 0, NORM_MINMAX);
	//rectified_map.convertTo(rectified_map, CV_8U, 1, 0);
	//imshow("disparity_retified", rectified_map);
	//imwrite("../data/disparity_retified.png", rectified_map);
	//cout << F;

	waitKey(0);
}