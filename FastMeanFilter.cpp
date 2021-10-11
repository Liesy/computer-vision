#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat img, processed;

void cal_integral(Mat &src, Mat &output) {
    //在积分图上方加一行0，左边加一列0，方便计算
    output = Mat::zeros(src.rows + 1, src.cols + 1, CV_32FC3);

    for (int row = 1; row < output.rows; row++) {
        float sum_row[3] = {0};
        for (int col = 1; col < output.cols; col++) {
            for (int c = 0; c < 3; c++) {
                //行累加,积分图的(1,1)对应原图(0,0),(i,j)对应(i-1,j-1)
                sum_row[c] += src.at<Vec3f>(row - 1, col - 1)[c];
                output.at<Vec3f>(row, col)[c] = output.at<Vec3f>(row - 1, col)[c] + sum_row[c];
            }
        }
    }
}

void FastMeanFilter(Mat &input, Mat &output, int window_size) {
    //转换格式
    Mat src, dst;
    input.convertTo(src, CV_32FC3);

    int rows = src.rows, cols = src.cols;
    dst = Mat::zeros(rows, cols, CV_32FC3);

    int border = window_size / 2;
    Mat src_border;
    copyMakeBorder(src, src_border, border, border, border, border, BORDER_REFLECT_101);

    //得到积分图
    Mat img_inte;
    cal_integral(src_border, img_inte);

    //均值滤波
    float mean = 0;
    float area = (float) window_size * (float) window_size;
    for (int row = border; row < rows + border; row++) {
        for (int col = border; col < cols + border; col++) {
            for (int c = 0; c < 3; c++) {
                //注意积分图索引要加1
                //注意计算时四个位置的行列与索引行列的对应关系
                float top_left = img_inte.at<Vec3f>(row - border - 1 + 1, col - border - 1 + 1)[c];
                float top_right = img_inte.at<Vec3f>(row - border - 1 + 1, col + border + 1)[c];
                float bottom_left = img_inte.at<Vec3f>(row + border + 1, col - border - 1 + 1)[c];
                float bottom_right = img_inte.at<Vec3f>(row + border + 1, col + border + 1)[c];

                mean = (bottom_right - top_right - bottom_left + top_left) / area;
                if (mean < 0) mean = 0;
                else if (mean > 255) mean = 255;

                dst.at<Vec3f>(row - border, col - border)[c] = mean;
            }
        }
    }

    dst.convertTo(output, img.type());
}

void windowSizeController(int val, void *) {
    if (val <= 1) imshow("filtering", img);
    else {
        //保证滤波窗口为奇数×奇数
        FastMeanFilter(img, processed, val % 2 != 0 ? val : val + 1);
        imshow("filtering", processed);
    }
}

void timeCompare() {
    double t, t1, t2;
    t = (double) getTickCount();
    FastMeanFilter(img, processed, 49);
    t1 = (double) getTickCount();
    boxFilter(img, processed, -1, Size(49, 49));
    t2 = (double) getTickCount();
    cout << "my program finished in " << ((t1 - t) * 1000.0) / getTickFrequency() << "ms" << '\n';
    cout << "boxFilter finished in " << ((t2 - t1) * 1000.0) / getTickFrequency() << "ms" << '\n';
}

int main() {
    ios::sync_with_stdio(false);

    img = imread("/home/liyang/图片/eileen.png");

    timeCompare();

    namedWindow("original img", WINDOW_AUTOSIZE);
    moveWindow("original img", 200, 200);
    imshow("original img", img);

    namedWindow("filtering", WINDOW_AUTOSIZE);
    moveWindow("filtering", 800, 150);

    createTrackbar("win size", "filtering", nullptr, 10, windowSizeController);
    windowSizeController(0, nullptr);

    waitKey(0);
    destroyAllWindows();

    return 0;
}