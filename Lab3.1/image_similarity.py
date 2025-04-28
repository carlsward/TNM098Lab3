"""
in /TNM098LAB3 RUN
python3 lab3.1/image_similarity.py
line 223 "reference_idx" variable changes ref image
"""
import cv2
import numpy as np
import os
from scipy.spatial import distance
from typing import List, Tuple
import matplotlib.pyplot as plt

class ImageAnalyzer:
    def __init__(self):
        self.feature_weights = {
            'color_hist': 0.35,    # Color histogram weight - highest as color is often the most important visual feature
            'edge_features': 0.30,  # Edge features weight - second highest as edges capture structural information
            'luminance': 0.15,     # Luminance distribution weight - medium importance for texture and contrast
            'central_region': 0.15, # Central region features weight - medium importance for main subject
            'corner_features': 0.05 # Corner features weight - lowest as corners are more sensitive to noise
        }

    def load_image(self, path: str) -> np.ndarray:
        """
        Load and preprocess image for analysis.
        - Resizes images to a consistent size (256x256) for uniform feature extraction
        - Ensures all images are processed with the same dimensions
        """
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        return cv2.resize(img, (256, 256))  # Resize for consistent analysis

    def extract_color_histogram(self, img: np.ndarray) -> np.ndarray:
        """
        Extract color features using HSV color space histogram.
        - Converts to HSV color space (better for color comparison than RGB)
        - Creates a 3D histogram with 12 bins for each channel (Hue, Saturation, Value)
        - Total of 12×12×12 = 1,728 bins for detailed color information
        - Normalizes the histogram to make it scale-invariant
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 12, 12], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def extract_edge_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extract structural features using edge detection.
        - Uses adaptive thresholding to handle varying lighting conditions
        - Applies Canny edge detection to find significant edges
        - Creates a 64-bin histogram of edge strengths
        - Captures the structural information and shapes in the image
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        edges = cv2.Canny(thresh, 50, 150)
        hist = cv2.calcHist([edges], [0], None, [64], [0, 256])
        return cv2.normalize(hist, hist).flatten()

    def extract_luminance_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extract texture and contrast features using gradient magnitude.
        - Calculates gradient magnitude using Sobel operators
        - Creates a 64-bin histogram of gradient magnitudes
        - Captures texture information and local contrast
        - Helps distinguish between smooth and textured regions
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        hist = cv2.calcHist([magnitude.astype(np.uint8)], [0], None, [64], [0, 256])
        return cv2.normalize(hist, hist).flatten()

    def extract_central_region(self, img: np.ndarray) -> np.ndarray:
        """
        Extract features from the central region of the image.
        - Focuses on the central 50% of the image
        - Creates a color histogram of the central region
        - Helps capture the main subject of the image
        - Useful for images where the main subject is centered
        """
        h, w = img.shape[:2]
        center = img[h//4:3*h//4, w//4:3*w//4]
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def extract_corner_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extract corner features using Harris corner detection.
        - Detects corners using Harris corner detection
        - Creates a 32-bin histogram of corner strengths
        - Captures structural details and keypoints
        - Useful for identifying specific points of interest
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        hist = cv2.calcHist([corners.astype(np.uint8)], [0], None, [32], [0, 256])
        return cv2.normalize(hist, hist).flatten()

    def extract_features(self, img: np.ndarray) -> np.ndarray:
        """
        Combine all features into a single feature vector.
        - Extracts all individual features
        - Applies the predefined weights to each feature type
        - Combines them into a single vector
        - Normalizes the final vector to unit length for cosine similarity
        """
        features = []
        
        # Extract all features
        color_hist = self.extract_color_histogram(img)
        edge_features = self.extract_edge_features(img)
        luminance = self.extract_luminance_features(img)
        central_region = self.extract_central_region(img)
        corner_features = self.extract_corner_features(img)
        
        # Combine features with weights
        features.extend(color_hist * self.feature_weights['color_hist'])
        features.extend(edge_features * self.feature_weights['edge_features'])
        features.extend(luminance * self.feature_weights['luminance'])
        features.extend(central_region * self.feature_weights['central_region'])
        features.extend(corner_features * self.feature_weights['corner_features'])
        
        # Normalize the combined feature vector
        features = np.array(features)
        return features / np.linalg.norm(features)

    def compute_similarity_matrix(self, image_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Compute similarity matrix for all images using cosine similarity.
        - Extracts features for all images
        - Computes pairwise similarity using cosine similarity
        - The distance is calculated as 1 - cosine_similarity
        - Returns a matrix where:
          * 0 means identical images
          * 1 means completely different images
          * Values in between indicate degree of similarity
        """
        n_images = len(image_paths)
        features = []
        
        # Extract features for all images
        for path in image_paths:
            img = self.load_image(path)
            features.append(self.extract_features(img))
        
        # Compute similarity matrix using cosine similarity
        similarity_matrix = np.zeros((n_images, n_images))
        for i in range(n_images):
            for j in range(n_images):
                # Cosine similarity: 1 - similarity = distance
                similarity_matrix[i, j] = 1 - np.dot(features[i], features[j])
        
        return similarity_matrix, image_paths

    def rank_similar_images(self, distance_matrix: np.ndarray, image_paths: List[str], 
                          reference_image_idx: int) -> List[Tuple[str, float]]:
        """Rank images by similarity to reference image"""
        distances = distance_matrix[reference_image_idx]
        ranked_images = list(zip(image_paths, distances))
        ranked_images.sort(key=lambda x: x[1])
        return ranked_images

    def visualize_similarity_matrix(self, distance_matrix: np.ndarray, image_paths: List[str]):
        """Visualize the similarity matrix"""
        plt.figure(figsize=(12, 10))
        plt.imshow(distance_matrix, cmap='viridis')
        plt.colorbar(label='Distance')
        
        # Set ticks and labels
        tick_labels = [os.path.basename(path) for path in image_paths]
        plt.xticks(range(len(image_paths)), tick_labels, rotation=45)
        plt.yticks(range(len(image_paths)), tick_labels)
        plt.title('Image Similarity Matrix')
        plt.tight_layout()
        plt.savefig('similarity_matrix.png')
        plt.close()

    def visualize_all_images(self, ranked_images: List[Tuple[str, float]], reference_image: str):
        """Visualize all images in a grid layout"""
        # Create a 4x3 grid for 12 images
        fig, axes = plt.subplots(4, 3, figsize=(15, 20))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        # Flatten the axes array for easier indexing
        axes = axes.flatten()
        
        # Show all images in order of similarity
        for i, (img_path, dist) in enumerate(ranked_images):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
            
            # Highlight the reference image
            if img_path == reference_image:
                axes[i].set_title(f'Reference Image\n{os.path.basename(img_path)}\nDistance: {dist:.4f}', 
                                color='red', fontweight='bold')
            else:
                axes[i].set_title(f'Rank {i}\n{os.path.basename(img_path)}\nDistance: {dist:.4f}')
            
            axes[i].axis('off')
        
        plt.suptitle('All Images Ranked by Similarity to Reference Image', fontsize=16, y=0.95)
        plt.savefig('all_images.png', bbox_inches='tight')
        plt.close()

def main():
    # Initialize analyzer
    analyzer = ImageAnalyzer()
    
    # Get all image paths
    image_dir = "lab3.1"
    image_paths = [os.path.join(image_dir, f"{i:02d}.jpg") for i in range(1, 13)]
    
    # Compute similarity matrix
    similarity_matrix, image_paths = analyzer.compute_similarity_matrix(image_paths)
    
    # Choose reference image (e.g., first image)
    reference_idx = 5
    ranked_images = analyzer.rank_similar_images(similarity_matrix, image_paths, reference_idx)
    
    # Print results
    print("\nRanking of images by similarity to reference image:")
    print(f"Reference image: {image_paths[reference_idx]}")
    print("\nRank\tImage\t\tDistance")
    print("-" * 40)
    for i, (img_path, dist) in enumerate(ranked_images):
        print(f"{i+1}\t{os.path.basename(img_path)}\t{dist:.4f}")
    
    # Create visualizations
    analyzer.visualize_similarity_matrix(similarity_matrix, image_paths)
    analyzer.visualize_all_images(ranked_images, image_paths[reference_idx])
    print("\nVisualizations saved as 'similarity_matrix.png' and 'all_images.png'")

if __name__ == "__main__":
    main() 