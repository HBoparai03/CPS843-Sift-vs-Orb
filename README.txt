# SIFT vs ORB Feature Matching Comparison

A small Python/OpenCV script that benchmarks **SIFT** vs **ORB** feature detection + matching on a reference image under common transformations:
- Rotation (30°)
- Scale (1.5×)
- Brightness (+50)
- Gaussian noise (σ=15)

It uses **BFMatcher (k-NN, k=2)** + **Lowe’s ratio test (0.75)** and prints a summary table with:
- keypoints (reference + transformed)
- good matches + match %
- runtime per method

## Requirements
- Python 3.x
- OpenCV (`opencv-contrib-python` recommended for SIFT)
- NumPy

Install:
```bash
pip install opencv-contrib-python numpy

##Usage
Put an image in the project folder (default: trike.jpg)
Run:
python feature_matching_comparison.py
