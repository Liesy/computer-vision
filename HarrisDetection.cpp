#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const char *img_path = "/home/liyang/图片/sunflower.png";

Mat src_img, src_gray;
Mat cornerStrength_1, cornerStrength_2;
Mat corner_img_1, corner_img_2;
int val;

void showImage(Mat &image, const char *windowName, int location_x, int location_y) {
    namedWindow(windowName, WINDOW_AUTOSIZE);
    moveWindow(windowName, location_x, location_y);
    imshow(windowName, image);
}

void harris(Mat &src, Mat &dst, int blockSize, int k_size, float k) {
    assert(src.type() == CV_8UC1 || src.type() == CV_32FC1);
    int rows = src.rows, cols = src.cols;
    dst = Mat::zeros(rows, cols, CV_32FC1);//matrix R

    Mat Ix, Iy;
    Sobel(src, Ix, CV_32FC1, 1, 0, k_size);
    Sobel(src, Iy, CV_32FC1, 0, 1, k_size);

    Mat Ix2 = Mat::zeros(rows, cols, CV_32FC1);
    Mat Iy2 = Mat::zeros(rows, cols, CV_32FC1);
    Mat Ixy = Mat::zeros(rows, cols, CV_32FC1);
    for (int y = blockSize / 2; y < rows - blockSize / 2; y++) {
        for (int x = blockSize / 2; x < cols - blockSize / 2; x++) {
            //a=Ix*Ix, b=Iy*Iy, c=Ix*Iy
            float a = 0, b = 0, c = 0;
            for (int wy = y - blockSize / 2; wy <= y + blockSize / 2; wy++) {
                for (int wx = x - blockSize / 2; wx <= x + blockSize / 2; wx++) {
                    float current_dx = Ix.at<float>(wy, wx), current_dy = Iy.at<float>(wy, wx);
                    a += current_dx * current_dx;
                    b += current_dy * current_dy;
                    c += current_dx * current_dy;
                }
            }
            Ix2.at<float>(y, x) = a;
            Iy2.at<float>(y, x) = b;
            Ixy.at<float>(y, x) = c;
            //trace=Ix2+Iy2, det=Ix2*Iy2-Ixy*Ixy
            float trace = a + b, det = a * b - c * c;
            dst.at<float>(y, x) = det - k * trace * trace;
        }
    }
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

void controller(int, void *) {
    ////my harris
    cornerDetection(src_img, corner_img_1, cornerStrength_1, (float) val);
    showImage(corner_img_1, "Output Image", 800, 200);
    ////opencv harris
    cornerDetection(src_img, corner_img_2, cornerStrength_2, (float) val);
    showImage(corner_img_2, "OpenCV Refer", 1400, 200);
}

int main() {
    ios::sync_with_stdio(false);

    src_img = imread(img_path);
    showImage(src_img, "Input Image", 200, 200);
    cvtColor(src_img, src_gray, COLOR_BGR2GRAY);//BGR != RGB

    double t, t1, t2;
    t = (double) getTickCount();
    harris(src_gray, cornerStrength_1, 3, 3, 0.04);
    t1 = (double) getTickCount();
    cornerHarris(src_gray, cornerStrength_2, 3, 3, 0.04);
    t2 = (double) getTickCount();
    cout << "my program finished in " << ((t1 - t) * 1000.0) / getTickFrequency() << "ms" << '\n';
    cout << "cornerHarris finished in " << ((t2 - t1) * 1000.0) / getTickFrequency() << "ms" << '\n';

    val = 100;//不同图像对应不同阈值，需要自己调
    createTrackbar("threshold", "Input Image", &val, 200, controller);
    controller(0, nullptr);

    waitKey(0);
    destroyAllWindows();

    return 0;
}