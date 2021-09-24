#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

double bilinear(int a, int b, int c, int d, double dx, double dy) {
    double h1 = a + dx * (b - a);
    double h2 = c + dx * (d - c);
    return h1 + dy * (h2 - h1);
}

Mat morphing(const Mat &src) {
    Mat processed = Mat::zeros(src.size(), src.type());
    int row = src.rows, col = src.cols;
    for (int x = 0; x < row; x++) {
        for (int y = 0; y < col; y++) {
            //坐标中心归一化
            double X = x / ((row - 1) / 2.0) - 1.0;
            double Y = y / ((col - 1) / 2.0) - 1.0;
            double r = sqrt(X * X + Y * Y);

            if (r >= 1) {
                for (int c = 0; c < 3; c++)
                    processed.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(src.at<Vec3b>(x, y)[c]);
            } else {
                double theta = X * X + Y * Y - 2.0 * sqrt(X * X + Y * Y) + 1.0;//(1-r)^2=r^2-2r+1
                double x_ = cos(theta) * X - sin(theta) * Y;
                double y_ = sin(theta) * X + cos(theta) * Y;
                //坐标还原
                x_ = (x_ + 1.0) * ((row - 1) / 2.0);
                y_ = (y_ + 1.0) * ((col - 1) / 2.0);

                if (x_ < 0 || y_ < 0 || x_ >= row || y >= col) {
                    for (int c = 0; c < 3; c++)
                        processed.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(0);
                } else {//双线性插值
                    int x1 = (int) x_, y1 = (int) y_;//左上角像素坐标
                    for (int c = 0; c < 3; c++) {
                        if (x1 == row - 1 || y1 == col - 1)
                            processed.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(src.at<Vec3b>(x1, y1)[c]);
                        else {
                            int aa = src.at<Vec3b>(x1, y1)[c];
                            int bb = src.at<Vec3b>(x1, y1 + 1)[c];
                            int cc = src.at<Vec3b>(x1 + 1, y1)[c];
                            int dd = src.at<Vec3b>(x1 + 1, y1 + 1)[c];
                            double dx = x_ - (double) x1;
                            double dy = y_ - (double) y1;
                            //  aa -------- cc
                            //  |  *(x_,y_) |
                            //  |           |
                            //  bb -------- dd
                            processed.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(bilinear(aa, bb, cc, dd, dx, dy));
                        }
                    }
                }
            }
        }
    }
    return processed;
}

int main() {
    ios::sync_with_stdio(false);

    Mat img = imread("/home/liyang/图片/mengnalisha.jpg");

    namedWindow("original img", WINDOW_AUTOSIZE);
    moveWindow("original img", 200, 200);
    imshow("original img", img);

    Mat imgAffine = morphing(img);

    namedWindow("image morphing", WINDOW_AUTOSIZE);
    moveWindow("image morphing", 800, 200);
    imshow("image morphing", imgAffine);

    waitKey(0);
    destroyAllWindows();

    return 0;
}