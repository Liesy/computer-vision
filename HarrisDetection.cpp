#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const char *img_path = "/home/liyang/图片/eileen.png";

void showImage(Mat &image, const char *windowName, int location_x, int location_y) {
    namedWindow(windowName, WINDOW_AUTOSIZE);
    moveWindow(windowName, location_x, location_y);
    imshow(windowName, image);
}

void harris(Mat &src, Mat &dst, int blockSize, int k_size, double k) {
    assert(src.type() == CV_8UC1 || src.type() == CV_32FC1);
    int rows = src.rows, cols = src.cols;
    dst = Mat::zeros(rows, cols, CV_32FC1);
}

void cornerDetection(Mat &src, Mat &dst, Mat &cornerStrength, float threshold) {
    assert(cornerStrength.type() == CV_32FC1);
    int rows = src.rows, cols = src.cols;
    dst = src.clone();

    Mat norm;
    normalize(cornerStrength, norm, 0, 255, NORM_MINMAX, CV_32FC1);

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            if (norm.at<float>(y, x) < threshold) continue;
            circle(dst, Point(x, y), 3, Scalar(0, 0, 255));
        }
    }
}

int main() {
    ios::sync_with_stdio(false);

    Mat src_img;
    src_img = imread(img_path);
    showImage(src_img, "Input Image", 200, 200);

    Mat src_gray;
    cvtColor(src_img, src_gray, COLOR_BGR2GRAY);//BGR != RGB

    Mat cornerStrength_1, cornerStrength_2;
    double t, t1, t2;
    t = (double) getTickCount();
    harris(src_gray, cornerStrength_1, 2, 3, 0.04);
    t1 = (double) getTickCount();
    cornerHarris(src_gray, cornerStrength_2, 2, 3, 0.04);
    t2 = (double) getTickCount();

    Mat corner_img_1, corner_img_2;
    cornerDetection(src_img, corner_img_1, cornerStrength_1, 88);
    cornerDetection(src_img, corner_img_2, cornerStrength_2, 88);
    showImage(corner_img_1, "Output Image", 800, 200);
    showImage(corner_img_2, "OpenCV Refer", 1400, 200);
    cout << "my program finished in " << ((t1 - t) * 1000.0) / getTickFrequency() << "ms" << '\n';
    cout << "cornerHarris finished in " << ((t2 - t1) * 1000.0) / getTickFrequency() << "ms" << '\n';

    waitKey(0);
    destroyAllWindows();

    return 0;
}