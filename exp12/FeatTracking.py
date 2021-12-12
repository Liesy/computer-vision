import numpy as np
import cv2 as cv
import os
import argparse

global screenshot, roi, target, bbox
global k_roi, d_roi
captured = False
blue, green, red = (255, 0, 0), (0, 255, 0), (0, 0, 255)


def show_image(img, win_name: str, location_x=0, location_y=0) -> None:
    if cv.getWindowProperty(win_name, cv.WND_PROP_AUTOSIZE) == -1:
        cv.namedWindow(win_name)
        cv.moveWindow(win_name, location_x, location_y)
    cv.imshow(win_name, img)


def merge_image(img1, img2):
    assert img1.shape == img2.shape
    row, col, c = img1.shape
    merge = np.zeros((row, col * 2, c), dtype='uint8')
    merge[:row, :col] = img1
    merge[:row, col:] = img2
    return merge


def feature_detect(img):
    def keypoint2point(keypoint):
        point = np.zeros(len(keypoint) * 2, np.float32)
        for i in range(len(keypoint)):
            point[i * 2] = keypoint[i].pt[0]
            point[i * 2 + 1] = keypoint[i].pt[1]
        point = point.reshape(-1, 2)
        return point

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if img.shape[-1] == 3 else img
    surf = cv.xfeatures2d.SURF_create(2000)
    key_pt, desc = surf.detectAndCompute(img_gray, None)
    return key_pt, keypoint2point(key_pt), desc


def get_match_image(img_left, kps_left, desc_left, img_right, kps_right, desc_right):
    matcher = cv.DescriptorMatcher_create("BruteForce")
    raw_matches = matcher.knnMatch(desc_left, desc_right, 2)
    good = []
    for m, n in raw_matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    return cv.drawMatches(img_left, kps_left, img_right, kps_right, good, None)


def target_tracking(file: str) -> None:
    def get_roi(img, box):
        ret = np.zeros(img.shape, dtype='uint8')
        p1, p2 = bbox2point(box)
        ret[p1[1]:p2[1], p1[0]:p2[0]] = img[p1[1]:p2[1], p1[0]:p2[0]]
        return ret

    def bbox2point(box):
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        return p1, p2

    global screenshot, roi, target, bbox, captured
    global k_roi, d_roi
    tracker = cv.TrackerKCF_create()
    video = cv.VideoCapture(file)
    pause_time = int(1000 / video.get(cv.CAP_PROP_FPS))
    while True:
        ok, frame = video.read()
        if not ok:
            break
        t = cv.getTickCount()
        if cv.waitKey(pause_time) == 27:  # press Esc to select target
            bbox = cv.selectROI('video', frame, False)
            pt1, pt2 = bbox2point(bbox)

            screenshot = frame
            cv.rectangle(screenshot, pt1, pt2, red, 2)
            roi = get_roi(frame, bbox)
            k_roi, _, d_roi = feature_detect(roi)
            target = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
            show_image(target, 'target')

            ok = tracker.init(frame, bbox)
            captured = True
        if not captured:
            show_image(frame, 'video')
            continue

        ok, bbox = tracker.update(frame)
        if ok:
            cv.rectangle(frame, bbox2point(bbox)[0], bbox2point(bbox)[1], blue, 2)
            roi_frame = get_roi(frame, bbox)
            k_frame, _, d_frame = feature_detect(roi_frame)
            merge = get_match_image(frame, k_frame, d_frame, screenshot, k_roi, d_roi)
        else:
            merge = merge_image(frame, screenshot)
        fps = cv.getTickFrequency() / (cv.getTickCount() - t)
        cv.putText(merge, f'FPS: {int(fps)}', (100, 50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        show_image(merge, 'video')


def main():
    video_path = os.path.join(args.video_dir, f'{args.video_name}.mp4')
    assert os.path.exists(video_path), 'file not exists.'

    cv.namedWindow('video')
    cv.moveWindow('video', 400, 200)
    target_tracking(video_path)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default='/home/liyang/视频')
    parser.add_argument('--video_name', type=str, default='AvA_bunny')
    args = parser.parse_args()
    print(args)

    main()
