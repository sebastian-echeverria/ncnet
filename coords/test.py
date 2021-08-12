import numpy as np
import cv2 as cv

from sift import find_matching_points
from sift import show_matches
from pix2coord import points_to_coordinates


def main():
    mosaic_path = '../datasets/mughal/Dataset/orthomosaic/nust.tif'
    template_path_prefix = '../datasets/mughal/Dataset/data/Image'

    src_pts = []
    dst_pts = []
    kp1 = None
    kp2 = None
    good_matches = None

    mode = "detect"
    total_images = 1
    starting = 11
    show_plot = True
    block_plot = True

    num_images_matched = 0
    for i in range(starting, starting+total_images):
        template_path = template_path_prefix + str(i) + ".jpg"
        print(f"Finding image: {template_path}")
        print(f"Matched so far: {num_images_matched}/{i} ({num_images_matched * 100 / i}%)")
        template = cv.imread(template_path, 0)
        mosaic = cv.imread(mosaic_path, 0)

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
            show_matches(template, mosaic, projected_corners, kp1, kp2, good_matches, matchesMask, "images/matching.png", block_plot)

    print(f"Stats: matched {num_images_matched} out of {i} ({num_images_matched * 100 / i}%)")


if __name__ == '__main__':
    main()
