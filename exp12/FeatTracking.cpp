#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

using namespace std;
using namespace cv;

const char *videoPath = "/home/liyang/视频/AvA_bunny.mp4";

Mat img, img_gray, img_pre, img_pre_gray, screenshot, screenshot_grey, target, merged;
Rect bbox, bbox_copy;//NOLINT 检测框
bool captured = false;

vector<Point2f> point1, point2, point_copy;
vector<uchar> status;
vector<float> err;

void showImage(Mat &image, const char *windowName, int location_x = 0, int location_y = 0) {
    //当窗口关闭时，getWindowProperty将返回-1
    if (getWindowProperty(windowName, WND_PROP_AUTOSIZE) == -1) {
        namedWindow(windowName, WINDOW_AUTOSIZE);
        moveWindow(windowName, location_x, location_y);
    }
    imshow(windowName, image);
}

void targetTracking() {
    Ptr<Tracker> tracker = TrackerKCF::create();
    VideoCapture video(videoPath);
    if (!video.isOpened()) {
        cout << "video open error." << '\n';
        return;
    }

    double fps = video.get(CAP_PROP_FPS); //获取视频帧率
    double pauseTime = 1000 / fps; //两幅画面中间间隔
    while (video.read(img)) {
        auto t = (double) getTickCount();
        if (waitKey(pauseTime) == 27) {//NOLINT Esc键按下暂停，选取目标
            bbox = selectROI("video", img, false);
            bbox_copy = bbox;
            cout << bbox.x << " " << bbox.y << " " << bbox.width << " " << bbox.height << '\n';
            target = img(bbox);

            img.copyTo(screenshot);
            rectangle(screenshot, bbox, Scalar(0, 0, 255), 2);
            tracker->init(screenshot, bbox);//跟踪器初始化

            img.copyTo(img_pre);
            cvtColor(img_pre, img_pre_gray, COLOR_BGR2GRAY);
            goodFeaturesToTrack(img_pre_gray(bbox), point1, 100, 0.01, 10, Mat());
            point_copy = point1;
            for (auto &p:point1) circle(screenshot(bbox), p, 1, Scalar(255, 0, 0), 2);

            imwrite("/home/liyang/视频/screenshot.png", screenshot);
            imwrite("/home/liyang/视频/target.png", target);
            captured = true;
        }
        if (!captured) {
            showImage(img, "video");
            continue;
        }
        if (tracker->update(img, bbox)) {
            rectangle(img, bbox, (0, 0, 255), 2);

            cvtColor(img, img_gray, COLOR_BGR2GRAY);
            calcOpticalFlowPyrLK(img_pre_gray, img_gray, point1, point2, status, err, Size(50, 50), 3);
            for (auto &p:point2) circle(img(bbox), p, 1, Scalar(255, 0, 0), 2);

            //合并截图和当前帧
            int rows = img.rows, cols = img.cols;
            merged.create(rows, cols * 2, img.type());
            img.copyTo(merged(Rect(0, 0, cols, rows)));
            screenshot.copyTo(merged(Rect(cols, 0, cols, rows)));
            auto gap = getTickFrequency() / ((double) getTickCount() - t);
            putText(merged, "FPS : " + to_string(int(gap)), Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50),
                    2);
            for (int i = 0; i < point2.size(); i++) {
                Point2f pt1(point2[i].x + bbox.x, point2[i].y + bbox.y), pt2(point_copy[i].x + cols + bbox_copy.x,
                                                                             point_copy[i].y + bbox_copy.y);
                line(merged, pt1, pt2, Scalar(0, 255, 0));
            }
            showImage(merged, "video");

            swap(point1, point2);
            img_pre_gray = img_gray.clone();
        } else {
            int rows = img.rows, cols = img.cols;
            merged.create(rows, cols * 2, img.type());
            img.copyTo(merged(Rect(0, 0, cols, rows)));
            screenshot.copyTo(merged(Rect(cols, 0, cols, rows)));
            auto gap = getTickFrequency() / ((double) getTickCount() - t);
            putText(merged, "FPS : " + to_string(int(gap)), Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50),
                    2);
            showImage(merged, "video");
        }
    }
    video.release();
}

int main() {
    ios::sync_with_stdio(false);

    namedWindow("video");
    moveWindow("video", 400, 200);

    targetTracking();

    waitKey(0);
    destroyAllWindows();

    return 0;
}