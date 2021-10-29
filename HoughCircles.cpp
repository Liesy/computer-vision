#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const char *img_path = "/home/liyang/study/cv/Circle Detection Dataset/100.jpg";

Mat img;

void circleDetection(Mat &src) {
    if (src.empty()) CV_Error(Error::StsNullPtr, "no src img");
    int rows = src.rows, cols = src.cols;
    Mat processed = Mat::zeros(rows, cols, src.type());

    Mat src_gray;
    cvtColor(src, src_gray, COLOR_RGB2GRAY);

    string win2 = "detection";
    namedWindow(win2, WINDOW_AUTOSIZE);
    moveWindow(win2, 800, 200);
    imshow(win2, processed);
}

int main() {
    ios::sync_with_stdio(false);

    img = imread(img_path);

    string win1 = "origin";
    namedWindow(win1, WINDOW_AUTOSIZE);
    moveWindow(win1, 200, 200);
    imshow(win1, img);

    circleDetection(img);

    waitKey(0);
    destroyAllWindows();

    return 0;
}