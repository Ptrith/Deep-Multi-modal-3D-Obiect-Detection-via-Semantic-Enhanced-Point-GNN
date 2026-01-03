"""This file implements PointPainting mechanism for semantic feature fusion.

PointPainting projects 2D semantic segmentation results onto 3D point clouds
to create semantically enriched point features.
"""

import numpy as np
import cv2
import tensorflow as tf


class PointPainter:
    """Class to perform PointPainting - projecting semantic scores onto points."""
    
    def __init__(self, num_classes=19):
        """
        Args:
            num_classes: Number of semantic classes from the segmentation model.
        """
        self.num_classes = num_classes
        
    def project_points_to_image(self, points_3d, calib):
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: [N, 3] numpy array of 3D points in camera coordinates
            calib: Calibration object with projection matrices
            
        Returns:
            points_2d: [N, 2] numpy array of pixel coordinates (u, v)
            valid_mask: [N] boolean mask for points within image bounds
        """
        # Convert to homogeneous coordinates
        points_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        
        # Project to image plane: p = K * [R|t] * P
        # Assuming calib.P2 is the projection matrix for left camera
        points_2d_homo = np.dot(points_homo, calib.P2.T)
        
        # Convert from homogeneous to pixel coordinates
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        # Create valid mask (within image bounds and in front of camera)
        valid_mask = (points_2d_homo[:, 2] > 0) & \
                     (points_2d[:, 0] >= 0) & (points_2d[:, 0] < calib.image_width) & \
                     (points_2d[:, 1] >= 0) & (points_2d[:, 1] < calib.image_height)
        
        return points_2d, valid_mask
    
    def paint_points(self, points_xyz, semantic_image, calib):
        """
        Paint 3D points with semantic scores from 2D segmentation.
        
        Args:
            points_xyz: [N, 3] numpy array of 3D points
            semantic_image: [H, W, C] numpy array of per-pixel semantic scores
            calib: Calibration object
            
        Returns:
            semantic_features: [N, C] numpy array of semantic scores for each point
        """
        # Project points to image
        points_2d, valid_mask = self.project_points_to_image(points_xyz, calib)
        
        # Initialize semantic features
        semantic_features = np.zeros((points_xyz.shape[0], self.num_classes), 
                                     dtype=np.float32)
        
        # Get valid points
        valid_points_2d = points_2d[valid_mask].astype(np.int32)
        
        # Clip to image boundaries (redundant check for safety)
        valid_points_2d[:, 0] = np.clip(valid_points_2d[:, 0], 0, 
                                         semantic_image.shape[1] - 1)
        valid_points_2d[:, 1] = np.clip(valid_points_2d[:, 1], 0, 
                                         semantic_image.shape[0] - 1)
        
        # Assign semantic scores to valid points
        semantic_features[valid_mask] = semantic_image[
            valid_points_2d[:, 1], valid_points_2d[:, 0], :]
        
        return semantic_features


class SemanticSegmentationModel:
    """Wrapper for semantic segmentation model (e.g., DeepLabV3+)."""
    
    def __init__(self, model_path=None, num_classes=19):
        """
        Args:
            model_path: Path to pretrained segmentation model
            num_classes: Number of output classes
        """
        self.num_classes = num_classes
        self.model_path = model_path
        self.model = None
        
        # Class mapping for common datasets
        # KITTI classes: background, car, pedestrian, cyclist, etc.
        self.class_names = [
            'background', 'car', 'truck', 'pedestrian', 'person_sitting',
            'cyclist', 'tram', 'misc', 'van', 'building', 'vegetation',
            'terrain', 'sky', 'road', 'sidewalk', 'fence', 'pole',
            'traffic_light', 'traffic_sign'
        ]
    
    def load_model(self):
        """Load pretrained semantic segmentation model."""
        if self.model_path is not None:
            # Load TensorFlow model
            # This is a placeholder - actual implementation depends on model format
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Loaded semantic segmentation model from {self.model_path}")
            except:
                print("Warning: Could not load segmentation model. Using dummy model.")
                self.model = None
        else:
            print("No segmentation model path provided. Using dummy segmentation.")
            self.model = None
    
    def predict(self, image):
        """
        Run semantic segmentation on image.
        
        Args:
            image: [H, W, 3] numpy array (RGB image)
            
        Returns:
            semantic_scores: [H, W, num_classes] numpy array of per-pixel scores
        """
        if self.model is not None:
            # Preprocess image
            input_image = self.preprocess_image(image)
            
            # Run inference
            logits = self.model.predict(input_image)
            
            # Convert logits to probabilities
            semantic_scores = tf.nn.softmax(logits, axis=-1).numpy()[0]
        else:
            # Dummy segmentation for testing without pretrained model
            semantic_scores = self._dummy_segmentation(image)
        
        return semantic_scores
    
    def preprocess_image(self, image):
        """Preprocess image for semantic segmentation model."""
        # Resize to model input size (e.g., 512x1024 for DeepLabV3+)
        resized = cv2.resize(image, (1024, 512))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def _dummy_segmentation(self, image):
        """
        Create dummy segmentation for testing purposes.
        Assigns simple heuristics based on image regions.
        """
        h, w = image.shape[:2]
        semantic_scores = np.zeros((h, w, self.num_classes), dtype=np.float32)
        
        # Simple heuristics:
        # - Lower part of image: road (class 13)
        # - Upper part: sky (class 12)
        # - Middle regions with certain colors: cars, pedestrians
        
        # Background as default
        semantic_scores[:, :, 0] = 0.3
        
        # Road at bottom
        road_mask = np.zeros((h, w), dtype=bool)
        road_mask[int(h*0.6):, :] = True
        semantic_scores[road_mask, 13] = 0.7
        
        # Sky at top
        sky_mask = np.zeros((h, w), dtype=bool)
        sky_mask[:int(h*0.3), :] = True
        semantic_scores[sky_mask, 12] = 0.8
        
        # Detect dark regions as potential vehicles
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            dark_mask = (gray < 100) & (~road_mask) & (~sky_mask)
            semantic_scores[dark_mask, 1] = 0.6  # car class
        
        # Normalize to sum to 1
        row_sums = semantic_scores.sum(axis=2, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        semantic_scores = semantic_scores / row_sums
        
        return semantic_scores


def augment_points_with_semantics(points_attr, semantic_features, 
                                   concat_mode='concat'):
    """
    Augment point features with semantic information.
    
    Args:
        points_attr: [N, M] original point attributes (e.g., intensity, RGB)
        semantic_features: [N, C] semantic scores
        concat_mode: 'concat' to concatenate, 'weighted' for attention fusion
        
    Returns:
        enhanced_features: [N, M+C] or [N, M] enhanced point features
    """
    if concat_mode == 'concat':
        # Simple concatenation
        enhanced_features = np.hstack([points_attr, semantic_features])
    elif concat_mode == 'weighted':
        # Use semantic confidence as attention weights (simplified)
        semantic_confidence = np.max(semantic_features, axis=1, keepdims=True)
        enhanced_features = points_attr * (1.0 + semantic_confidence)
        enhanced_features = np.hstack([enhanced_features, semantic_features])
    else:
        raise ValueError(f"Unknown concat_mode: {concat_mode}")
    
    return enhanced_features.astype(np.float32)

