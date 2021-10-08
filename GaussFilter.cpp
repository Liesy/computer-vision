#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat img, processed;
int sigma1, sigma2;

void Gaussian(const Mat &input, Mat &output, double sigma) {
    Mat input_border;
    copyMakeBorder(input, input_border, 50, 50, 50, 50, BORDER_REFLECT_101);
}

void sigmaController(int, void *) {
    double sigma = 1.0 * sigma1 + 0.01 * sigma2;
    Gaussian(img, processed, sigma);
    imshow("filtering", processed);
}

int main() {
    ios::sync_with_stdio(false);

    img = imread("/home/liyang/图片/mengnalisha.jpg");
    processed = Mat::zeros(img.size(), img.type());

    namedWindow("original img", WINDOW_AUTOSIZE);
    moveWindow("original img", 200, 200);
    imshow("original img", img);

    namedWindow("filtering", WINDOW_AUTOSIZE);
    moveWindow("filtering", 800, 100);

    createTrackbar("integer part   ", "filtering", &sigma1, 100, sigmaController);
    createTrackbar("fractional part", "filtering", &sigma2, 99, sigmaController);

    waitKey(0);
    destroyAllWindows();

    return 0;
}