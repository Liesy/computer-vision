#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

int SigmoidControl;
Mat original, processed;

void ContrastControl(int, void *) {
    for (int y = 0; y < original.rows; y++)
        for (int x = 0; x < original.cols; x++)
            for (int c = 0; c < 3; c++) {
                double u = ((original.at<Vec3b>(y, x)[c] - 127.0) / 255.00) * SigmoidControl * 0.1;
                processed.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(
                        original.at<Vec3b>(y, x)[c] * (1.00 / (1.00 + exp(-u))) + 0.4);
            }
    imshow("exp1-1_original", original);
    imshow("exp1-1_processed", processed);
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false);

    original = imread("/home/liyang/图片/einstein.jpeg");
    if (original.empty()) return -1;
    processed = Mat::zeros(original.size(), original.type());

    SigmoidControl = 100;

    namedWindow("exp1-1_original", WINDOW_AUTOSIZE);
    moveWindow("exp1-1_original", 200, 200);
    namedWindow("exp1-1_processed", WINDOW_AUTOSIZE);
    moveWindow("exp1-1_processed", 800, 200);
    createTrackbar("Contrast Controller", "exp1-1_processed", &SigmoidControl, 100, ContrastControl);
    ContrastControl(0, nullptr);

    waitKey(0);

    return 0;
}