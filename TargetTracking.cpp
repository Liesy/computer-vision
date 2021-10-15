#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const char *videoPath = "";

Mat img, show_img, target_img;
bool flag_leftDown = false, tar_captured = false;
Point pointStart, pointEnd;//NOLINT 矩形框的起点/终点

void onMouse(int event, int x, int y, int flags, void *param) {
    if (event == EVENT_LBUTTONDOWN) {// left mouse button is pressed
        pointStart = Point(x, y);
        pointEnd = pointStart;
        flag_leftDown = true;
    } else if (flag_leftDown && event == EVENT_MOUSEMOVE) {
        img.copyTo(show_img);
        pointEnd = Point(x, y);
        if (pointStart != pointEnd) rectangle(show_img, pointStart, pointEnd, Scalar(255, 0, 0), 2);
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

        //两种情况下不在原始视频图像上刷新矩形
        //1. 起点等于终点
        //2. 左键按下且未抬起
        if (pointStart != pointEnd && !flag_leftDown)
            rectangle(img, pointStart, pointEnd, Scalar(255, 0, 0), 2);

        imshow("origin", img);

        if (tar_captured) {
            destroyWindow("origin");
            break;
        }
    }

    video.release();
}

void calHistogram(const Mat &src) {

}

double compareHistogram(const Mat &srcHist, const Mat &compHist) {

}

void tracking() {

}

int main() {
    ios::sync_with_stdio(false);

    namedWindow("origin", WINDOW_AUTOSIZE);
    moveWindow("origin", 400, 200);
    setMouseCallback("origin", onMouse);

    getTarget();
    calHistogram(target_img);

//    namedWindow("tracking", WINDOW_AUTOSIZE);
//    moveWindow("tracking", 1050, 200);

    waitKey(0);
    destroyAllWindows();

    return 0;
}