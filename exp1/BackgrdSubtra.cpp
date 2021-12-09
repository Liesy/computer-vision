#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int control = 100;
Mat img_1, img_2, processed;

void BackgroundSubtra(int, void *) {
    double diff;
    for (int y = 0; y < img_1.rows; y++) {
        for (int x = 0; x < img_1.cols; x++) {
            diff = 0.0;
            for (int c = 0; c < 3; c++)
                diff += pow((img_1.at<Vec3b>(y, x)[c] - img_2.at<Vec3b>(y, x)[c]), 2);
            diff = sqrt(diff);
            if (diff >= control) {
                for (int c = 0; c < 3; c++) {
                    processed.at<Vec3b>(y, x)[c] = 255;
                }
            } else {
                for (int c = 0; c < 3; c++) {
                    processed.at<Vec3b>(y, x)[c] = 0;
                }
            }
        }
    }
    imshow("background subtraction", processed);
}

int main(int argc, char **argv) {
    ios::sync_with_stdio(false);

    img_1 = imread("/home/liyang/study/cv/bgs/02.jpg");//original image
    img_2 = imread("/home/liyang/study/cv/bgs/02_bg.jpg");//background image

    //original image must be equal to background image in the field of rows and cols
    if (img_1.empty() || img_2.empty() || img_1.rows != img_2.rows || img_1.cols != img_2.cols) {
        cout << "error" << '\n';
        return -1;
    }

    processed = Mat::zeros(img_1.size(), img_1.type());

    namedWindow("original");
    moveWindow("original", 200, 200);
    namedWindow("background");
    moveWindow("background", 800, 200);

    imshow("original", img_1);
    imshow("background", img_2);

    namedWindow("background subtraction");
    moveWindow("background subtraction", 1400, 200);

    createTrackbar("control", "background subtraction", &control, 255, BackgroundSubtra);
    BackgroundSubtra(0, nullptr);

    waitKey(0);
    return 0;
}