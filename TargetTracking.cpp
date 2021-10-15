#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const char *videoPath = "/home/liyang/视频/diana.mp4";

Mat img, show_img, target_img;
bool flag_leftDown = false, tar_captured = false;
Point pointStart, pointEnd;//NOLINT 矩形框的起点/终点
double target_Hist[256][3] = {0};

void onMouse(int event, int x, int y, int flags, void *param) {
    if (event == EVENT_LBUTTONDOWN) {// left mouse button is pressed
        pointStart = Point(x, y);
        pointEnd = pointStart;
        flag_leftDown = true;
    } else if (flag_leftDown && event == EVENT_MOUSEMOVE) {
        img.copyTo(show_img);
        pointEnd = Point(x, y);
        if (pointStart != pointEnd) rectangle(show_img, pointStart, pointEnd, Scalar(0, 0, 255), 2);
        imshow("origin", show_img);
    } else if (flag_leftDown && event == EVENT_LBUTTONUP) {// left mouse button is released
        target_img = img(Rect(pointStart, pointEnd));//获取目标图像
        imshow("target", target_img);
        tar_captured = true;
        flag_leftDown = false;

        namedWindow("target", WINDOW_AUTOSIZE);
        moveWindow("target", 100, 200);
        imshow("target", target_img);
    }
}

void getTarget() {
    VideoCapture video(videoPath);
    if (!video.isOpened()) {
        cout << "video not open" << '\n';
        return;
    }

    double fps = video.get(CAP_PROP_FPS); //获取视频帧率
    double pauseTime = 1000 / fps; //两幅画面中间间隔

    while (true) {
        if (!flag_leftDown) video >> img;//左键没有按下，采取播放视频，否则暂停
        if (!img.data || waitKey(pauseTime) == 27) break;//NOLINT 图像为空或Esc键按下退出播放

        //两种情况下在原始视频图像上刷新矩形
        //1. 起点不等于终点
        //2. 左键按下且未抬起
        if (pointStart != pointEnd && !flag_leftDown)
            rectangle(img, pointStart, pointEnd, Scalar(0, 0, 255), 2);

        imshow("origin", img);

        if (tar_captured) {
            destroyWindow("origin");
            break;
        }
    }

    video.release();
}

void calHistogram(const Mat &src, double dst[256][3]) {
    //分离三个通道像素值
    vector<Mat> channels;
    split(src, channels);
    Mat B = channels.at(0);
    Mat G = channels.at(1);
    Mat R = channels.at(2);

    int rows = src.rows, cols = src.cols;
    double area = 1.0 * rows * cols;

    for (int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            ++dst[B.at<uchar>(y, x)][0];
            ++dst[G.at<uchar>(y, x)][1];
            ++dst[R.at<uchar>(y, x)][2];
        }
    }

    //直方图归一化
    for (int i = 0; i < 256; i++) dst[i][0] /= area, dst[i][1] /= area, dst[i][2] /= area;
}

void drawHistogram(const double hist[256][3]) {
    double BMax = 0, GMax = 0, RMax = 0;//记录归一化后最大的值，用于画图缩放
    for (int i = 0; i < 256; i++) {
        if (hist[i][0] > BMax) BMax = hist[i][0];
        if (hist[i][1] > GMax) GMax = hist[i][1];
        if (hist[i][2] > RMax) RMax = hist[i][2];
    }

    Mat hist_img(400, 800, CV_8UC3, Scalar(0, 0, 0));

    Point pt1, pt2;
    pt1.x = 0, pt2.x = 0;
    pt1.y = 400;

    double scale = 0;

    //绘制R通道直方图
    scale = 400 / RMax;
    for (int i = 0; i < 256; i++) {
        pt2.y = pt1.y - (int) (hist[i][2] * scale);
        line(hist_img, pt1, pt2, Scalar(0, 0, 255), 1, 8, 0);
        pt2.x = ++pt1.x;
    }
    //绘制G通道直方图
    scale = 400 / GMax;
    for (int i = 0; i < 256; i++) {
        pt2.y = pt1.y - (int) (hist[i][1] * scale);
        line(hist_img, pt1, pt2, Scalar(0, 255, 0), 1, 8, 0);
        pt2.x = ++pt1.x;
    }
    //绘制B通道直方图
    scale = 400 / BMax;
    for (int i = 0; i < 256; i++) {
        pt2.y = pt1.y - (int) (hist[i][0] * scale);
        line(hist_img, pt1, pt2, Scalar(255, 0, 0), 1, 8, 0);
        pt2.x = ++pt1.x;
    }

    namedWindow("Histogram", WINDOW_AUTOSIZE);
    moveWindow("Histogram", 400, 200);
    imshow("Histogram", hist_img);
}

double compareHistogram(const Mat &compImg) {
    double temp[256][3] = {0};
    calHistogram(compImg, temp);

    double res[3] = {0};
    for (int i = 0; i < 256; i++) {
        res[0] += sqrt(target_Hist[i][0] * temp[i][0]);
        res[1] += sqrt(target_Hist[i][1] * temp[i][1]);
        res[2] += sqrt(target_Hist[i][2] * temp[i][2]);
    }

    return ((res[0] + res[1] + res[2]) / 3);
}

void tracking() {
    int tar_cols = abs(pointEnd.x - pointStart.x);
    int tar_rows = abs(pointEnd.y - pointStart.y);
    cout << "target size = " << tar_rows << "*" << tar_cols << '\n';

    //目标搜索区域设定为原区域的周围且面积为原来三倍
    int X1 = pointStart.x - tar_cols;
    int X2 = pointStart.x + tar_cols;
    int Y1 = pointStart.y - tar_rows;
    int Y2 = pointStart.y + tar_rows;
    //越界检查
    X1 = X1 < 0 ? 0 : X1;
    Y1 = Y1 < 0 ? 0 : Y1;

    Point preStart;
    Point preEnd;

    Point get1(0, 0);
    Point get2(0, 0);

    VideoCapture video(videoPath);
    if (!video.isOpened()) {
        cout << "video not open.error!" << std::endl;
        return;
    }

    double fps = video.get(CAP_PROP_FPS); //获取视频帧率
    double pauseTime = 1000 / fps; //两幅画面中间间隔

    int w = static_cast<int>(video.get(CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(video.get(CAP_PROP_FRAME_HEIGHT));
    VideoWriter write;
    int codec = VideoWriter::fourcc('X', '2', '6', '4');
    write.open("/home/liyang/视频/diana_detection.avi", codec, fps, Size(w, h), true);

//    namedWindow("tracking", WINDOW_AUTOSIZE);
//    moveWindow("tracking", 1250, 200);

    while (true) {
        video >> img;
        if (!img.data || waitKey(pauseTime) == 27) break;//NOLINT 图像为空或Esc键按下退出播放

        double minDist = 1.0;//直方图对比的相似值

        for (int srh_y = Y1; srh_y <= Y2; srh_y += 10) {
            for (preStart.x = X1, preStart.y = srh_y; preStart.x <= X2; preStart.x += 10) {
                //初始化搜索区域
                if (preStart.x + tar_cols < img.cols) preEnd.x = preStart.x + tar_cols;
                else preEnd.x = img.cols - 1;
                if (preStart.y + tar_rows < img.rows) preEnd.y = preStart.y + tar_rows;
                else preEnd.y = img.rows - 1;

                Mat compareImg = img(Rect(preStart, preEnd));
                double ret = compareHistogram(compareImg);
                if (minDist > ret) {
                    get1 = preStart;
                    get2 = preEnd;
                    minDist = ret;
                }
            }
        }

        //在原始视频图像上刷新矩形，只有当与目标直方图很相似时才更新起点搜索区域，满足目标进行移动的场景
        if (minDist < 0.5) {
            X1 = get1.x - tar_cols;
            X2 = get1.x + tar_cols;
            Y1 = get1.y - tar_rows;
            Y2 = get1.y + tar_rows;
            X1 = X1 < 0 ? 0 : X1;
            Y1 = Y1 < 0 ? 0 : Y1;
        }
        if (minDist < 0.7)
            rectangle(img, get1, get2, Scalar(0, 0, 255), 2);

        //写入一帧
        write.write(img);
//        imshow("tracking", img);
    }

    video.release();
    write.release();
}

int main() {
    ios::sync_with_stdio(false);

    namedWindow("origin", WINDOW_AUTOSIZE);
    moveWindow("origin", 400, 200);
    setMouseCallback("origin", onMouse);

    getTarget();
    calHistogram(target_img, target_Hist);
    drawHistogram(target_Hist);
    tracking();

    waitKey(0);
    destroyAllWindows();

    return 0;
}