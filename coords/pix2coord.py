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

