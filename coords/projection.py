import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


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

    MIN_MATCH_COUNT = 10
    if len(good_matches) < MIN_MATCH_COUNT:
        raise Exception("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    return src_pts, dst_pts, kp1, kp2, good_matches


def project_first_in_second(img1, src_pts, dst_pts):
    """Projects the first image into the second one using the homography matrix from the matching points."""
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    print(f"Image shape: {img1.shape}")
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)

    return dst, matchesMask


def show_matches(img1, img2, dst, kp1, kp2, good, matchesMask):
    # Draw the projection of the first image into the second.
    img3 = cv.polylines(img2, [np.int32(dst)], True, 0, 3, cv.LINE_AA)

    # Add the keypoint matches between the images.
    draw_params = dict(matchColor=(0, 255, 0),   # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    if kp1 is not None and kp2 is not None and good is not None:
        img3 = cv.drawMatches(img1, kp1, img3, kp2, good, None, **draw_params)

    plt.imshow(img3, 'gray')
    plt.show()


def main():
    mosaic = cv.imread('../datasets/mughal/Dataset/orthomosaic/nust.tif', 0)
    template = cv.imread('../datasets/mughal/Dataset/data/Image11.jpg', 0)
    src_pts = np.array([[70.18304443, 147.3605804],
               [100.9704361, 190.8684692],
               [284.1752625, 26.5521946],
               [79.10481262, 53.83478546],
               [318.4112244, 120.8362427],
               [107.0935516, 182.6598358],
               [150.0068207, 186.5870361],
               [93.61287689, 164.9320679],
               [270.988678, 191.6297302],
               [75.79307556, 109.5082626],
               [237.7413483, 44.42756653],
               [237.7413483, 44.42756653],
               [284.3021851, 21.67831993],
               [239.8641663, 32.49781036],
               [239.8641663, 32.49781036],
               [39.91210556, 159.6648407],
               [126.5273132, 67.86650848],
               [65.35128021, 82.53801727]])
    dst_pts = np.array([[381.9416504, 345.0480652],
               [388.4534912, 230.4508972],
               [398.2503052, 226.7294312],
               [432.5168152, 219.4615784],
               [434.5153503, 319.7545776],
               [434.5153503, 319.7545776],
               [437.5349121, 347.0392456],
               [440.7446289, 322.4544373],
               [440.7446289, 322.4544373],
               [443.4916382, 256.9204712],
               [448.2037048, 220.2913208],
               [454.809906, 268.3821106],
               [458.9595337, 270.3993835],
               [467.2735596, 222.5155029],
               [468.8685608, 252.1152649],
               [470.7114563, 257.572998],
               [470.8974915, 219.303772],
               [470.8974915, 219.303772]])

    kp1 = None
    kp2 = None
    good_matches = None
    mode = "read"
    if mode == "detect":
        try:
            src_pts, dst_pts, kp1, kp2, good_matches = find_matching_points(template, mosaic)
        except Exception as ex:
            print(f"Error: {str(ex)}")
            exit(0)

    dest_loc, matchesMask = project_first_in_second(template, src_pts, dst_pts)
    print(dest_loc)
    show_matches(template, mosaic, dest_loc, kp1, kp2, good_matches, matchesMask)


if __name__ == '__main__':
    main()
