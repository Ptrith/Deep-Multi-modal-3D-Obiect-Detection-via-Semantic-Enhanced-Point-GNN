"""This file implements enhanced NMS with semantic confidence re-scoring.

The enhanced NMS combines geometric IoU with semantic confidence scores
to improve detection accuracy, especially for small objects like pedestrians.
"""

import numpy as np
from models.nms import (overlapped_boxes_3d_fast_poly, boxes_3d_to_corners,
                        bboxes_sort)


def rescore_boxes_with_semantics(box_labels, detection_boxes_3d, 
                                 detection_scores, semantic_scores,
                                 semantic_weight=0.3):
    """
    Re-score detection boxes by incorporating semantic confidence.
    
    Args:
        box_labels: [N] class labels for each box
        detection_boxes_3d: [N, 7] 3D box parameters
        detection_scores: [N] detection confidence scores
        semantic_scores: [N] semantic confidence scores from PointPainting
        semantic_weight: weight for semantic confidence (0-1)
        
    Returns:
        rescored_scores: [N] re-weighted confidence scores
    """
    # Combine detection score with semantic score
    # Formula: final_score = det_score * (1 + semantic_weight * sem_score)
    rescored_scores = detection_scores * (1.0 + semantic_weight * semantic_scores)
    
    # Clip to valid range [0, 1]
    rescored_scores = np.clip(rescored_scores, 0.0, 1.0)
    
    return rescored_scores


def adaptive_nms_threshold(box_labels, size_thresholds=None):
    """
    Get adaptive NMS thresholds for different object sizes.
    
    Args:
        box_labels: [N] class labels
        size_thresholds: dict mapping class names to IoU thresholds
        
    Returns:
        thresholds: [N] IoU thresholds for each box
    """
    if size_thresholds is None:
        # Default thresholds: lower for small objects (pedestrians)
        size_thresholds = {
            0: 0.1,  # Background
            1: 0.7,  # Car (large object, higher threshold)
            2: 0.7,  # Car variant
            3: 0.5,  # Pedestrian (small object, lower threshold)
            4: 0.5,  # Pedestrian variant
            5: 0.5,  # Cyclist
            6: 0.5,  # Cyclist variant
            7: 0.1,  # DontCare
        }
    
    # Get threshold for each box based on its class
    thresholds = np.array([size_thresholds.get(label, 0.5) 
                          for label in box_labels])
    
    return thresholds


def nms_boxes_3d_semantic(class_labels, detection_boxes_3d, detection_scores,
                          semantic_scores=None,
                          overlapped_fn=overlapped_boxes_3d_fast_poly,
                          overlapped_thres=0.1,
                          appr_factor=100.0,
                          top_k=-1,
                          attributes=None,
                          use_adaptive_threshold=True,
                          semantic_rescore_weight=0.3):
    """
    Semantic-aware 3D NMS with adaptive thresholds and semantic re-scoring.
    
    Args:
        class_labels: [N] class labels
        detection_boxes_3d: [N, 7] 3D boxes
        detection_scores: [N] detection scores
        semantic_scores: [N] semantic confidence scores (optional)
        overlapped_fn: function to compute box overlap
        overlapped_thres: base IoU threshold
        appr_factor: approximation factor for box corners
        top_k: keep top k boxes (-1 for all)
        attributes: additional attributes to track
        use_adaptive_threshold: use class-specific thresholds
        semantic_rescore_weight: weight for semantic re-scoring
        
    Returns:
        Tuple of (filtered_labels, filtered_boxes, filtered_scores, filtered_indices)
    """
    # Re-score boxes with semantic information if available
    if semantic_scores is not None:
        original_scores = detection_scores.copy()
        detection_scores = rescore_boxes_with_semantics(
            class_labels, detection_boxes_3d, detection_scores, 
            semantic_scores, semantic_rescore_weight)
    
    # Get adaptive thresholds if enabled
    if use_adaptive_threshold:
        adaptive_thresholds = adaptive_nms_threshold(class_labels)
    else:
        adaptive_thresholds = np.full_like(class_labels, overlapped_thres, 
                                          dtype=np.float32)
    
    # Sort by scores
    class_labels, detection_scores, detection_boxes_3d, attributes = \
        bboxes_sort(class_labels, detection_scores, detection_boxes_3d,
                   top_k=top_k, attributes=attributes)
    
    # Update adaptive thresholds after sorting
    if use_adaptive_threshold:
        adaptive_thresholds = adaptive_nms_threshold(class_labels)
    
    # Convert boxes to corners
    detection_boxes_3d_corners = boxes_3d_to_corners(
        detection_boxes_3d * appr_factor)
    
    # Perform NMS
    keep_indices = []
    classes = np.unique(class_labels)
    
    for cls in classes:
        if cls == 0:  # Skip background
            continue
        
        cls_mask = (class_labels == cls)
        cls_indices = np.where(cls_mask)[0]
        
        if len(cls_indices) == 0:
            continue
        
        cls_boxes_corners = detection_boxes_3d_corners[cls_indices]
        cls_scores = detection_scores[cls_indices]
        cls_thresholds = adaptive_thresholds[cls_indices]
        
        # NMS for this class
        cls_keep = []
        remaining = list(range(len(cls_indices)))
        
        while len(remaining) > 0:
            # Pick box with highest score
            idx = remaining[0]
            cls_keep.append(idx)
            
            if len(remaining) == 1:
                break
            
            # Compute overlap with remaining boxes
            current_box = cls_boxes_corners[idx]
            remaining_boxes = cls_boxes_corners[remaining[1:]]
            
            overlaps = overlapped_fn(current_box, remaining_boxes)
            
            # Use adaptive threshold for current box
            current_threshold = cls_thresholds[idx]
            
            # Keep boxes with low overlap
            keep_mask = overlaps <= current_threshold
            remaining = [remaining[i+1] for i, keep in enumerate(keep_mask) if keep]
        
        # Convert to original indices
        cls_keep_indices = cls_indices[cls_keep]
        keep_indices.extend(cls_keep_indices.tolist())
    
    keep_indices = np.array(keep_indices, dtype=np.int32)
    
    # Sort by original order
    if len(keep_indices) > 0:
        keep_indices = np.sort(keep_indices)
    
    # Filter results
    filtered_labels = class_labels[keep_indices]
    filtered_boxes = detection_boxes_3d[keep_indices]
    filtered_scores = detection_scores[keep_indices]
    
    if attributes is not None:
        filtered_attributes = attributes[keep_indices]
    else:
        filtered_attributes = keep_indices
    
    return filtered_labels, filtered_boxes, filtered_scores, filtered_attributes


def nms_boxes_3d_semantic_merge(class_labels, detection_boxes_3d, 
                                detection_scores, semantic_scores=None,
                                overlapped_fn=overlapped_boxes_3d_fast_poly,
                                overlapped_thres=0.1,
                                appr_factor=100.0,
                                top_k=-1,
                                attributes=None,
                                use_adaptive_threshold=True,
                                semantic_rescore_weight=0.3):
    """
    Semantic-aware 3D NMS with box merging and uncertainty scoring.
    
    Similar to nms_boxes_3d_uncertainty but with semantic enhancements.
    """
    # Re-score with semantics
    if semantic_scores is not None:
        detection_scores = rescore_boxes_with_semantics(
            class_labels, detection_boxes_3d, detection_scores,
            semantic_scores, semantic_rescore_weight)
    
    # Get adaptive thresholds
    if use_adaptive_threshold:
        adaptive_thresholds = adaptive_nms_threshold(class_labels)
    else:
        adaptive_thresholds = np.full_like(class_labels, overlapped_thres,
                                          dtype=np.float32)
    
    # Sort by scores
    class_labels, detection_scores, detection_boxes_3d, attributes = \
        bboxes_sort(class_labels, detection_scores, detection_boxes_3d,
                   top_k=top_k, attributes=attributes)
    
    # Update thresholds after sorting
    if use_adaptive_threshold:
        adaptive_thresholds = adaptive_nms_threshold(class_labels)
    
    # Convert to corners
    detection_boxes_3d_corners = boxes_3d_to_corners(
        detection_boxes_3d * appr_factor)
    
    # Merge boxes
    merged_indices = []
    merged_boxes = []
    merged_scores = []
    merged_labels = []
    
    classes = np.unique(class_labels)
    
    for cls in classes:
        if cls == 0:  # Skip background
            continue
        
        cls_mask = (class_labels == cls)
        cls_indices = np.where(cls_mask)[0]
        
        if len(cls_indices) == 0:
            continue
        
        cls_boxes = detection_boxes_3d[cls_indices]
        cls_boxes_corners = detection_boxes_3d_corners[cls_indices]
        cls_scores = detection_scores[cls_indices]
        cls_thresholds = adaptive_thresholds[cls_indices]
        
        remaining = list(range(len(cls_indices)))
        
        while len(remaining) > 0:
            idx = remaining[0]
            current_box = cls_boxes_corners[idx]
            current_threshold = cls_thresholds[idx]
            
            # Find overlapping boxes
            if len(remaining) > 1:
                overlaps = overlapped_fn(current_box, 
                                        cls_boxes_corners[remaining[1:]])
                overlap_mask = overlaps > current_threshold
                overlap_indices = [remaining[0]] + \
                                [remaining[i+1] for i, keep in enumerate(overlap_mask) if keep]
            else:
                overlap_indices = [remaining[0]]
            
            # Merge overlapping boxes (weighted average by score)
            overlap_boxes = cls_boxes[overlap_indices]
            overlap_scores = cls_scores[overlap_indices]
            
            if len(overlap_indices) > 1:
                # Weighted average
                weights = overlap_scores / np.sum(overlap_scores)
                merged_box = np.sum(overlap_boxes * weights[:, np.newaxis], axis=0)
                merged_score = np.mean(overlap_scores)  # or np.max
            else:
                merged_box = overlap_boxes[0]
                merged_score = overlap_scores[0]
            
            merged_boxes.append(merged_box)
            merged_scores.append(merged_score)
            merged_labels.append(cls)
            merged_indices.append(cls_indices[idx])
            
            # Remove merged boxes from remaining
            remaining = [r for r in remaining if r not in overlap_indices]
    
    if len(merged_boxes) == 0:
        return (np.array([], dtype=np.int32), 
                np.array([]).reshape(0, 7),
                np.array([]),
                np.array([], dtype=np.int32))
    
    merged_labels = np.array(merged_labels, dtype=np.int32)
    merged_boxes = np.array(merged_boxes)
    merged_scores = np.array(merged_scores)
    merged_indices = np.array(merged_indices, dtype=np.int32)
    
    return merged_labels, merged_boxes, merged_scores, merged_indices

