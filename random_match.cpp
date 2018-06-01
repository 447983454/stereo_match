#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <time.h>

using namespace std;
using namespace cv;

void init(Mat&dst, int r, int c, int maxOffset, int psize) {

	int half_psize = psize / 2;
	int offset = c - psize;
	CV_Assert(maxOffset + half_psize< 255);

	srand(unsigned(time(NULL)));
	//dst.create(r, c, CV_32F);
	dst = Mat::zeros(r, c, CV_32F);
	for (int y = half_psize; y + half_psize< r; y++) {
		float*row = dst.ptr<float>(y);
		for (int x = half_psize; x + half_psize < c; x++) {

			int range = min(maxOffset, x - half_psize);
			int temp = range==0?0:rand() % range;
			if (temp < 0)
				cout << "here";
			row[x] = temp;
		}
	}
}
float calc_score(Mat&imgL, Mat&imgR, int x, int y, int offset, int psize) {
	int half_psize = psize / 2;
	//if (x - half_psize < 0 || y - half_psize < 0 || x - offset - half_psize < 0 || x + half_psize >= imgL.cols || y + half_psize >= imgL.rows)
	if (x - half_psize < 0 || y - half_psize < 0 || x - offset - half_psize < 0
		||x+half_psize>=imgL.cols||y+half_psize>=imgL.rows)
		return FLT_MAX;
	Mat left = imgL(Rect(x - half_psize, y - half_psize, psize, psize));
	Mat right = imgR(Rect(x - offset - half_psize, y - half_psize, psize, psize));
	Mat result = left - right;
	result = result.mul(result);
	return sum(result)[0];
}
int choose_best(float new_score[3]) {
	//0 for curr, 1 for new_1, 2 for new_2
	if (new_score[0] < new_score[1])
		if (new_score[0] < new_score[2])
			return 0;
		else
			return 2;
	else if (new_score[1] < new_score[2])
		return 1;
	else
		return 2;
}
void update(Mat&imgL, Mat&imgR, Mat&score_mat, Mat&offset_mat, int maxOffset, int psize, bool inverse) {
	int half_psize = psize / 2;
	int x_start, x_end, y_start, y_end, step;
	if (inverse) {
		x_start = imgL.cols - half_psize;
		x_end = half_psize;
		y_start = imgL.rows - half_psize;
		y_end = half_psize;
		step = -1;
	}
	else {
		x_start = half_psize;
		x_end = imgL.cols - half_psize;
		y_start = half_psize;
		y_end = imgL.rows - half_psize;
		step = 1;
	}
	for (int y = y_start; y != y_end; y+=step) {
		float* row_score = score_mat.ptr<float>(y);
		float* row_offset = offset_mat.ptr<float>(y);
		float* y_row_offset = offset_mat.ptr<float>(y - step);
		//float*y_row_score = score_mat.ptr<float>(y - step);

		for (int x = x_start; x != x_end; x += step) {
			float new_score[3];
			new_score[0]= row_score[x];
			int new_offset[3];
			new_offset[0] = row_offset[x];
			if (x - step >= half_psize && x - step < imgL.cols-half_psize) {
				new_offset[1] = row_offset[x - step] + step;
				if (new_offset[1] > maxOffset)
					new_offset[1] = maxOffset;
				new_score[1] = calc_score(imgL, imgR, x, y, new_offset[1], psize);
			}
			else {
				new_offset[1] = FLT_MAX;
				new_score[1] = FLT_MAX;
			}
			if (y - step >= half_psize && y - step < imgL.rows - half_psize) {
				new_offset[2] = y_row_offset[x];
				if(new_offset[2]>maxOffset)
					new_offset[2] = maxOffset;
				new_score[2] = calc_score(imgL, imgR, x, y, new_offset[2], psize);
			}
			else {
				new_offset[2] = FLT_MAX;
				new_score[2] = FLT_MAX;
			}
			int choise = choose_best(new_score);
			row_offset[x] = new_offset[choise];
		}
	}
	//for (int y = half_psize; y + half_psize < imgL.rows; y++) {
	//	float* row_score = score_mat.ptr<float>(y);
	//	float* row_offset = offset_mat.ptr<float>(y); 
	//	for (int x = start; x != end; x += step) {
	//		int radius = imgL.cols/2;
	//		while (radius > 1) {
	//			int min_x = max(x - radius, half_psize);
	//			int max_x = min(imgL.cols - half_psize, x + radius);
	//			int random_x = rand()%(max_x-min_x)+min_x;
	//			float curr_score = row_score[x]; 
	//			int new_offset = row_offset[random_x] + x - random_x;
	//			if (new_offset > maxOffset)
	//				new_offset = maxOffset;
	//			float new_score = calc_score(imgL, imgR, x, y, new_offset, psize);
	//			if (curr_score > new_score) {
	//				row_offset[x] = new_offset;
	//				row_score[x] = new_score;
	//			}
	//			radius =radius / 4;
	//		}
	//	}
	//}
}
void patchmatch(Mat&imgL, Mat&imgR, Mat&dst, int maxOffset, int psize = 5, int n_iter = 3) {
	int r = imgL.rows, c = imgL.cols;
	init(dst, r, c, maxOffset, psize);
	Mat score_mat(r, c, CV_32F);
	int half_psize = psize / 2;
	for (int y = half_psize; y + half_psize < r; y++) {
		float*score_row = score_mat.ptr<float>(y);
		float*dst_row = dst.ptr<float>(y);
		for (int x = half_psize; x + half_psize < c; x++) {
			if (x == 81 && y == 182)
				cout << "here";
			score_row[x] = calc_score(imgL, imgR, x, y, dst_row[x], psize);
		}
	}
	//cout << "here";
	bool inverse = false;
	for (int count = 0; count < n_iter; count++) {
		update(imgL, imgR, score_mat, dst, maxOffset, psize, inverse);
		inverse = !inverse;
	}
}
int main() {
	Mat imgL = imread("../data/im2.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgR = imread("../data/im6.png", CV_LOAD_IMAGE_GRAYSCALE);
	imgL.convertTo(imgL, CV_32F);
	imgR.convertTo(imgR, CV_32F);
	Mat dst;
	patchmatch(imgL, imgR, dst, (imgL.cols / 8 + 15)&-16, 5, 3);
	normalize(dst, dst, 255, 0, NORM_MINMAX);
	dst.convertTo(dst, CV_8U);
	imshow("dst", dst);
	imwrite("random.png", dst);
	waitKey(0);
}