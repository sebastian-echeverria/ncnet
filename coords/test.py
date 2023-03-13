import os
import os.path
import sys

import numpy as np
import cv2 as cv

from sift import find_matching_points
from sift import show_matches
from pix2coord import points_to_coordinates


def main():
    mosaic_path = sys.argv[1]
    image_folder = sys.argv[2]
    color = False
    if len(sys.argv) > 3:
        color = True

    src_pts = []
    dst_pts = []
    kp1 = None
    kp2 = None
    good_matches = None

    mode = "detect"
    show_plot = True
    block_plot = False

    cv_color = 0
    out_color = 'gray'
    if color:
        cv_color = 1
        out_color = 'viridis'

    num_images_matched = 0
    i = 0
    for image in sorted(os.listdir(image_folder)):
        i = i + 1
        template_path = os.path.join(image_folder, image)
        print(f"Finding image: {template_path}")
        print(f"Matched so far: {num_images_matched}/{i} ({num_images_matched * 100 / i}%)")
        template = cv.imread(template_path, cv_color)
        mosaic = cv.imread(mosaic_path, cv_color)

        if mode == "detect":
            try:
                src_pts, dst_pts, kp1, kp2, good_matches = find_matching_points(template, mosaic)
                num_images_matched += 1
            except Exception as ex:
                print(f"Error: {str(ex)}")
                continue

        # Get GPS from set of matching points.
        gps_coords, projected_corners, matchesMask = points_to_coordinates(template_path, mosaic_path, src_pts, dst_pts)

        # Show matches, as well as KPs
        if show_plot:
            show_matches(template, mosaic, projected_corners, kp1, kp2, good_matches, matchesMask, "images/matching" + str(i) + ".png", block_plot, out_color)

    print(f"Stats: matched {num_images_matched} out of {i} ({num_images_matched * 100 / i}%)")


if __name__ == '__main__':
    main()
