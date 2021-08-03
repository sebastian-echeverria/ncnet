# Projection and Pixel to Coordinates

This code uses a list of points that correspond to matching points between two images (small and orthomosaic) as an input, and
 - sift.py: Uses homography (through OpenCV) to project the first image into the second one (orthomosaic)
 - pix2coord.py: Uses the GeoTIFF coordinate info in the orthomosaic to calculate the (lat, long) coordinates of the centroid of the projected image

The test.py code file runs a simple test where it can either use sift to find matches and then show the results, or use some hardcoded test points to test this out.

Dependencies:
 - Non-dockerized:
    - OpenCV
    - GDAL
    - Python packages in requirements.txt
        - `pip install -r requirements.txt`
 - Dockerized:
    - Docker

To run this:
 - Non-dockerized: `python test.py`
 - Dockerized version:
    - `bash build_container.sh`
    - `bash run_container.sh`
