#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const char *img_path = "/home/liyang/图片/horse_mask.png";

Mat img;

void connected(Mat &src) {//8连通的快速连通域算法
    int rows = src.rows, cols = src.cols;
    Mat processed = Mat::zeros(src.size(), src.type());

    int label = 0;
    for (int y = 1; y < rows - 1; y++) {
        for (int x = 1; x < cols - 1; x++) {
            if (src.at<uchar>(y, x) == 255) {
                //pass
            }
        }
    }

    string win2 = "connected component";
    namedWindow(win2, WINDOW_AUTOSIZE);
    moveWindow(win2, 1015, 200);
    imshow(win2, processed);
}

void distanceField(Mat &src) {//距离场可视化

}

int main() {
    ios::sync_with_stdio(false);

    img = imread(img_path, CV_8UC1);
    for (int y = 0; y < img.rows; y++) {//binary
        for (int x = 0; x < img.cols; x++) {
            img.at<uchar>(y, x) = img.at<uchar>(y, x) <= 127 ? 0 : 255;
        }
    }

    string win1 = "origin";
    namedWindow(win1, WINDOW_AUTOSIZE);
    moveWindow(win1, 0, 200);
    imshow(win1, img);

    connected(img);

    waitKey(0);
    destroyAllWindows();

    return 0;
}