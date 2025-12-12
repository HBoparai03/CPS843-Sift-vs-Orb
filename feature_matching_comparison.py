import cv2
import numpy as np
import time
from typing import List, Dict, Tuple


def generate_transformations(img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Generate transformed versions of the input image.
    
    Args:
        img: Input grayscale image
        
    Returns:
        List of tuples (transformation_name, transformed_image)
    """
    transformations = []
    h, w = img.shape
    
    # 1. Rotation (30 degrees)
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 30, 1.0)
    rotated = cv2.warpAffine(img, rotation_matrix, (w, h))
    transformations.append(("Rotation (30°)", rotated))
    
    # 2. Scale (1.5x)
    scaled = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    # Crop to original size for fair comparison
    sh, sw = scaled.shape
    start_y, start_x = (sh - h) // 2, (sw - w) // 2
    scaled = scaled[start_y:start_y+h, start_x:start_x+w]
    transformations.append(("Scale (1.5x)", scaled))
    
    # 3. Brightness adjustment (+50)
    brightened = cv2.add(img, 50)
    brightened = np.clip(brightened, 0, 255).astype(np.uint8)
    transformations.append(("Brightness (+50)", brightened))
    
    # 4. Noise (Gaussian noise)
    noise = np.random.normal(0, 15, img.shape).astype(np.float32)
    noisy = cv2.add(img.astype(np.float32), noise)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    transformations.append(("Gaussian Noise (σ=15)", noisy))
    
    return transformations


def match_features_sift(img1: np.ndarray, img2: np.ndarray) -> Tuple[int, int, float, List]:
    """
    Detect and match features using SIFT.
    
    Args:
        img1: Reference image
        img2: Transformed image
        
    Returns:
        Tuple of (num_keypoints_ref, num_keypoints_trans, runtime, good_matches)
    """
    start_time = time.time()
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # Create BFMatcher with L2 norm for SIFT
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # Match descriptors using k-nearest neighbors (k=2 for Lowe's ratio test)
    if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test (ratio = 0.75)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
    else:
        good_matches = []
    
    runtime = time.time() - start_time
    
    return len(kp1), len(kp2), runtime, good_matches


def match_features_orb(img1: np.ndarray, img2: np.ndarray) -> Tuple[int, int, float, List]:
    """
    Detect and match features using ORB.
    
    Args:
        img1: Reference image
        img2: Transformed image
        
    Returns:
        Tuple of (num_keypoints_ref, num_keypoints_trans, runtime, good_matches)
    """
    start_time = time.time()
    
    # Create ORB detector (capped at 1000 features)
    orb = cv2.ORB_create(nfeatures=15000)
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Create BFMatcher with Hamming distance for ORB
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Match descriptors using k-nearest neighbors (k=2 for Lowe's ratio test)
    if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test (ratio = 0.75)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
    else:
        good_matches = []
    
    runtime = time.time() - start_time
    
    return len(kp1), len(kp2), runtime, good_matches


def visualize_matches(img1: np.ndarray, img2: np.ndarray, 
                     kp1: List, kp2: List, matches: List, 
                     title: str = "Feature Matches"):
    """
    Visualize matched features between two images.
    
    Args:
        img1: Reference image
        img2: Transformed image
        kp1: Keypoints from image 1
        kp2: Keypoints from image 2
        matches: List of good matches
        title: Window title
    """
    # Draw matches
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Display
    cv2.imshow(title, img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Main function to run the feature matching comparison."""
    
    # Load reference image (grayscale)
    ref_image_path = 'trike.jpg'  # Change this to your image path
    ref_img = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
    
    if ref_img is None:
        print(f"Error: Could not load image '{ref_image_path}'")
        print("Please ensure the image file exists in the current directory.")
        return
    
    print(f"Loaded reference image: {ref_image_path}")
    print(f"Image size: {ref_img.shape}")
    print("\nGenerating transformations...")
    
    # Generate transformed images
    transformations = generate_transformations(ref_img)
    
    # Initialize results storage
    results: List[Dict] = []
    
    print("\n" + "="*80)
    print("FEATURE MATCHING COMPARISON: SIFT vs ORB")
    print("="*80)
    
    # Process each transformation
    for trans_name, trans_img in transformations:
        print(f"\nProcessing: {trans_name}")
        
        # SIFT matching
        kp1_sift, kp2_sift, time_sift, matches_sift = match_features_sift(ref_img, trans_img)
        match_percentage_sift = (len(matches_sift) / kp1_sift * 100) if kp1_sift > 0 else 0.0
        print(f"  SIFT - Keypoints: {kp1_sift} (ref) / {kp2_sift} (trans), "
              f"Good matches: {len(matches_sift)} ({match_percentage_sift:.2f}%), Time: {time_sift:.4f}s")
        
        # ORB matching
        kp1_orb, kp2_orb, time_orb, matches_orb = match_features_orb(ref_img, trans_img)
        match_percentage_orb = (len(matches_orb) / kp1_orb * 100) if kp1_orb > 0 else 0.0
        print(f"  ORB  - Keypoints: {kp1_orb} (ref) / {kp2_orb} (trans), "
              f"Good matches: {len(matches_orb)} ({match_percentage_orb:.2f}%), Time: {time_orb:.4f}s")
        
        # Store results
        results.append({
            'transformation': trans_name,
            'method': 'SIFT',
            'keypoints_ref': kp1_sift,
            'keypoints_trans': kp2_sift,
            'good_matches': len(matches_sift),
            'match_percentage': match_percentage_sift,
            'runtime': time_sift
        })
        
        results.append({
            'transformation': trans_name,
            'method': 'ORB',
            'keypoints_ref': kp1_orb,
            'keypoints_trans': kp2_orb,
            'good_matches': len(matches_orb),
            'match_percentage': match_percentage_orb,
            'runtime': time_orb
        })
    
    # Display summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Transformation':<25} {'Method':<8} {'KP (Ref)':<10} {'KP (Trans)':<12} "
          f"{'Good Matches':<14} {'Match %':<10} {'Runtime (s)':<12}")
    print("-"*80)
    
    for result in results:
        print(f"{result['transformation']:<25} {result['method']:<8} "
              f"{result['keypoints_ref']:<10} {result['keypoints_trans']:<12} "
              f"{result['good_matches']:<14} {result['match_percentage']:<10.2f} "
              f"{result['runtime']:<12.4f}")
    
    print("="*80)
    
    # Optional: Visualize matches (uncomment to enable)
    # print("\nVisualizing matches...")
    # for trans_name, trans_img in transformations:
    #     # SIFT visualization
    #     sift = cv2.SIFT_create()
    #     kp1, des1 = sift.detectAndCompute(ref_img, None)
    #     kp2, des2 = sift.detectAndCompute(trans_img, None)
    #     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    #     if des1 is not None and des2 is not None and len(des1) > 1 and len(des2) > 1:
    #         matches = bf.knnMatch(des1, des2, k=2)
    #         good_matches = [m for match_pair in matches if len(match_pair) == 2 
    #                        for m, n in [match_pair] if m.distance < 0.75 * n.distance]
    #         visualize_matches(ref_img, trans_img, kp1, kp2, good_matches, 
    #                         f"SIFT: {trans_name}")


if __name__ == "__main__":
    main()

