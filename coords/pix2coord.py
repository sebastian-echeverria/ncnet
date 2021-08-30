import numpy as np
import cv2 as cv

from osgeo import gdal
from osgeo import osr


class CoordinateConversor():
    """Converts coordinates from a GeoTIFF base image."""

    def __init__(self, mosaic_path):
        mosaic = gdal.Open(mosaic_path)
        self.c, self.a, self.b, self.f, self.d, self.e = mosaic.GetGeoTransform()

        srs = osr.SpatialReference()
        srs.ImportFromWkt(mosaic.GetProjection())
        srsLatLong = srs.CloneGeogCS()
        self.coord_transform = osr.CoordinateTransformation(srs, srsLatLong)

    def pixel_to_coord(self, col, row):
        """Returns global coordinates to pixel center using base-0 raster index"""
        xp = self.a * col + self.b * row + self.c
        yp = self.d * col + self.e * row + self.f
        coords = self.coord_transform.TransformPoint(xp, yp)
        return coords

    @staticmethod
    def calculate_centroid(points):
        """Calculates the centroid of a given set of points."""
        points = np.int32(points)
        M = cv.moments(points)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return cx, cy


def project_first_in_second(img1, src_pts, dst_pts):
    """Projects the first image into the second one using the homography matrix from the matching points."""
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    print(f"Image shape: {img1.shape}")
    h, w = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)

    return dst, matchesMask


def check_if_rectangular_like(pts, centroid):
    """Checks if the points form a rectangle-ish shape."""
    # First calculate distances between every corner and centroid.
    dists = np.zeros(4)
    for i in range(0, 4):
        dists[i] = np.linalg.norm(pts[i] - centroid)
    print(f"Distances: {dists}")

    # Now calculate percentual differences between the previous distances. If there is a percentual difference
    # higher than the threshold, it means the projections has a non-right-angle-ish corner.
    ANGLE_PERCENT_THRESHOLD = 20
    bad_shape = False
    diffsp = np.zeros(4)
    for i in range(0, 4):
        diffsp[i] = abs(dists[i] - dists[(i+1)%4])/dists[i]*100
        if diffsp[i] > ANGLE_PERCENT_THRESHOLD:
            bad_shape = True
    print(f"% Diffs: {diffsp}")

    if bad_shape:
        print("BAD SHAPE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return False
    else:
        return True


def points_to_coordinates(template_path, mosaic_path, src_pts, dst_pts):
    """Gets two images and two sets of matching points, and returns the projected corners of the image, and the GPS
    coordinates of its centroid."""
    template = cv.imread(template_path, 0)
    projected_corners, matchesMask = project_first_in_second(template, src_pts, dst_pts)
    print(f"Projection: {projected_corners}")

    centroid = CoordinateConversor.calculate_centroid(projected_corners)
    print(f"Center: {centroid}")
    check_if_rectangular_like(projected_corners, centroid)

    conversor = CoordinateConversor(mosaic_path)
    gps_coords = conversor.pixel_to_coord(centroid[0], centroid[1])
    print(f"GPS coords: {gps_coords}")

    return gps_coords, projected_corners, matchesMask
