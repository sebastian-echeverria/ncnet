import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pathlib import Path
import os


def find_matching_points(img1, img2):
    """Returns two sets of matching points between the two given images."""

    # Find the keypoints and descriptors with SIFT
    print("Finding keypoints with SIFT", flush=True)
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Find matches using FLANN.
    print("Finding matches using FLANN", flush=True)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    MIN_MATCH_COUNT = 8
    if len(good_matches) < MIN_MATCH_COUNT:
        raise Exception("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
    else:
        print(f"Found {len(good_matches)} matches")

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return src_pts, dst_pts, kp1, kp2, good_matches


def show_matches(img1, img2, dst, kp1, kp2, good, matchesMask, output_file, block_plot=False):
    # Draw the projection of the first image into the second.
    img3 = cv.polylines(img2, [np.int32(dst)], True, 0, 3, cv.LINE_AA)

    # Add the keypoint matches between the images.
    draw_params = dict(matchColor=(0, 255, 0),   # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    if kp1 is not None and kp2 is not None and good is not None:
        img3 = cv.drawMatches(img1, kp1, img3, kp2, good, None, **draw_params)

    plt.clf()
    plt.imshow(img3, 'gray')

    # Save to file
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file)

    if block_plot:
        plt.show()
    else:
        plt.draw()
        plt.pause(0.01)
