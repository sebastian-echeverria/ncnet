import numpy as np
import cv2 as cv

from sift import find_matching_points
from sift import show_matches
from pix2coord import project_first_in_second
from pix2coord import CoordinateConversor


def main():
    mosaic_path = '../datasets/mughal/Dataset/orthomosaic/nust.tif'
    mosaic = cv.imread(mosaic_path, 0)
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
    mode = "detect"
    if mode == "detect":
        try:
            src_pts, dst_pts, kp1, kp2, good_matches = find_matching_points(template, mosaic)
        except Exception as ex:
            print(f"Error: {str(ex)}")
            exit(0)

    dest_loc, matchesMask = project_first_in_second(template, src_pts, dst_pts)
    print(f"Projection: {dest_loc}")

    centroid = CoordinateConversor.calculate_centroid(dest_loc)
    print(f"Center: {centroid}")

    conversor = CoordinateConversor(mosaic_path)
    gps_coords = conversor.pixel_to_coord(centroid[0], centroid[1])
    print(f"GPS coords: {gps_coords}")

    show_matches(template, mosaic, dest_loc, kp1, kp2, good_matches, matchesMask, "images/matching.png")


if __name__ == '__main__':
    main()