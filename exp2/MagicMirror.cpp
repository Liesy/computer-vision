#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

double bilinear(int a, int b, int c, int d, double dx, double dy) {
    double h1 = a + dx * (b - a);
    double h2 = c + dx * (d - c);
    return h1 + dy * (h2 - h1);
}

Mat magic_mirror(const Mat &src) {
    Mat processed = Mat::zeros(src.size(), src.type());
    int height = src.rows, width = src.cols;
    Point center(width / 2, height / 2);
    double R = sqrt(width * width + height * height) / 2.0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double r = norm(Point(x, y) - center);//获得当前点到中心点的距离
            if (r >= R) {
                for (int c = 0; c < 3; c++)
                    processed.at<Vec3b>(y, x)[c] = src.at<Vec3b>(y, x)[c];
            } else {
                double x_ = (x - center.x) * r / R + center.x;
                double y_ = (y - center.y) * r / R + center.y;
                int x1 = (int) x_, y1 = (int) y_;//左上角像素坐标
                for (int c = 0; c < 3; c++) {
                    if (x1 == width - 1 || y1 == height - 1)
                        processed.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(src.at<Vec3b>(y1, x1)[c]);
                    else {
                        int aa = src.at<Vec3b>(y1, x1)[c];
                        int bb = src.at<Vec3b>(y1, x1 + 1)[c];
                        int cc = src.at<Vec3b>(y1 + 1, x1)[c];
                        int dd = src.at<Vec3b>(y1 + 1, x1 + 1)[c];
                        double dx = x_ - (double) x1;
                        double dy = y_ - (double) y1;
                        processed.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(bilinear(aa, bb, cc, dd, dx, dy));
                    }
                }
            }
        }
    }
    return processed;
}

int main() {
    ios::sync_with_stdio(false);

    VideoCapture cap(CAP_ANY);
    Mat frame;

    namedWindow("magic mirror", WINDOW_AUTOSIZE);
    moveWindow("magic mirror", 200, 200);

    while (true) {
        cap >> frame;
        if (!frame.empty()) {
            imshow("magic mirror", magic_mirror(frame));
        } else break;
        if (waitKey(20) == 'q') break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}