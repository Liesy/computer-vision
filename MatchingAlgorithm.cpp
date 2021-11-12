#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const char *img_path_original = "/home/liyang/图片/sunflower_original.png";
const char *img_path_scale = "/home/liyang/图片/sunflower_scale.png";
const char *img_path_local = "/home/liyang/图片/sunflower.png";

Mat src_img_original, src_gray_original;
Mat src_img_scale, src_gray_scale;
Mat src_img_local, src_gray_local;

int numFeatures;

void showImage(Mat &image, const char *windowName, int location_x, int location_y) {
    if (getWindowProperty(windowName, WND_PROP_AUTOSIZE) == -1)//当窗口关闭时，getWindowProperty将返回-1
        namedWindow(windowName, WINDOW_AUTOSIZE);
    moveWindow(windowName, location_x, location_y);
    imshow(windowName, image);
}

vector<DMatch> matchPtsFilter(vector<DMatch> &matchPts) {
    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    double minDist = 1e+5;
    for (auto &pt:matchPts) {
        if (pt.distance < minDist) minDist = pt.distance;
    }
    double threshold = max(2 * minDist, 30.0);
    vector<DMatch> ret;
    for (auto &pt:matchPts) {
        if (pt.distance <= threshold) ret.emplace_back(pt);
    }
    return ret;
}

void siftMatching() {
    double start, end;

    //SIFT特征点提取
    Ptr<FeatureDetector> siftDetector = SIFT::create(numFeatures);
    vector<KeyPoint> KP_original, KP_scale, KP_local;
    start = (double) getTickCount();
    siftDetector->detect(src_gray_original, KP_original);
    end = (double) getTickCount();
    cout << "SIFT finished in " << ((end - start) * 1000.0) / getTickFrequency() << "ms" << '\n';
    siftDetector->detect(src_gray_scale, KP_scale);
    siftDetector->detect(src_gray_local, KP_local);

    //特征点描述
    Ptr<DescriptorExtractor> siftDescriptor = SIFT::create(numFeatures);
    Mat desc_original, desc_scale, desc_local;
    siftDescriptor->compute(src_gray_original, KP_original, desc_original);
    siftDescriptor->compute(src_gray_scale, KP_scale, desc_scale);
    siftDescriptor->compute(src_gray_local, KP_local, desc_local);

    //获得匹配特征点
    FlannBasedMatcher scaleMatcher, localMatcher;
    vector<DMatch> scaleMatchPts, localMatchPts;
    scaleMatcher.match(desc_original, desc_scale, scaleMatchPts);
    localMatcher.match(desc_original, desc_local, localMatchPts);

    Mat img_scaleMatch, img_localMatch;
    drawMatches(src_img_original, KP_original, src_img_scale, KP_scale, matchPtsFilter(scaleMatchPts), img_scaleMatch);
    drawMatches(src_img_original, KP_original, src_img_local, KP_local, matchPtsFilter(localMatchPts), img_localMatch);

    showImage(img_localMatch, "sift local matching", 0, 160);
    showImage(img_scaleMatch, "sift scale matching", 200, 190);
}

void orbMatching() {
    double start, end;

    //SIFT特征点提取
    Ptr<FeatureDetector> orbDetector = ORB::create(numFeatures);
    vector<KeyPoint> KP_original, KP_scale, KP_local;
    start = (double) getTickCount();
    orbDetector->detect(src_gray_original, KP_original);
    end = (double) getTickCount();
    cout << "orb finished in " << ((end - start) * 1000.0) / getTickFrequency() << "ms" << '\n';
    orbDetector->detect(src_gray_scale, KP_scale);
    orbDetector->detect(src_gray_local, KP_local);

    //特征点描述
    Ptr<DescriptorExtractor> orbDescriptor = ORB::create(numFeatures);
    Mat desc_original, desc_scale, desc_local;
    orbDescriptor->compute(src_gray_original, KP_original, desc_original);
    orbDescriptor->compute(src_gray_scale, KP_scale, desc_scale);
    orbDescriptor->compute(src_gray_local, KP_local, desc_local);

    //获得匹配特征点
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    vector<DMatch> scaleMatchPts, localMatchPts;
    matcher->match(desc_original, desc_scale, scaleMatchPts);
    matcher->match(desc_original, desc_local, localMatchPts);

    Mat img_scaleMatch, img_localMatch;
    drawMatches(src_img_original, KP_original, src_img_scale, KP_scale, matchPtsFilter(scaleMatchPts), img_scaleMatch);
    drawMatches(src_img_original, KP_original, src_img_local, KP_local, matchPtsFilter(localMatchPts), img_localMatch);

    showImage(img_localMatch, "orb local matching", 400, 220);
    showImage(img_scaleMatch, "orb scale matching", 800, 250);
}

void controller(int, void *) {
    siftMatching();
    orbMatching();
}

int main() {
    ios::sync_with_stdio(false);

    src_img_original = imread(img_path_original);
    src_img_scale = imread(img_path_scale);
    src_img_local = imread(img_path_local);
    cvtColor(src_img_original, src_gray_original, COLOR_BGR2GRAY);//BGR != RGB
    cvtColor(src_img_scale, src_gray_scale, COLOR_BGR2GRAY);
    cvtColor(src_img_local, src_gray_local, COLOR_BGR2GRAY);

    numFeatures = 1000;
    namedWindow("controller", WINDOW_AUTOSIZE);
    moveWindow("controller", 0, 0);
    createTrackbar("nfeatures", "controller", &numFeatures, 1000, controller);
    controller(0, nullptr);

    waitKey(0);
    destroyAllWindows();

    return 0;
}