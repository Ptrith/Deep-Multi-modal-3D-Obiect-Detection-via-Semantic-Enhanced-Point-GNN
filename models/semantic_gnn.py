"""This file implements semantic-aware GNN layers with semantic consistency.

These layers enhance the original Point-GNN by incorporating semantic similarity
into edge feature calculations.
"""

import tensorflow as tf
import numpy as np
try:
    import tensorflow.contrib.slim as slim
except (ImportError, AttributeError):
    import tf_slim as slim
from models.gnn import (multi_layer_neural_network_fn, graph_scatter_max_fn,
                        normalization_fn_dict, activation_fn_dict)


def compute_semantic_similarity(src_semantic, dst_semantic, mode='cosine'):
    """
    Compute semantic similarity between source and destination vertices.
    
    Args:
        src_semantic: [N_edges, C] semantic features of source vertices
        dst_semantic: [N_edges, C] semantic features of destination vertices
        mode: 'cosine', 'l2', or 'kl' for similarity metric
        
    Returns:
        similarity: [N_edges, 1] similarity scores (higher = more similar)
    """
    if mode == 'cosine':
        # Cosine similarity
        src_norm = tf.nn.l2_normalize(src_semantic, axis=1)
        dst_norm = tf.nn.l2_normalize(dst_semantic, axis=1)
        similarity = tf.reduce_sum(src_norm * dst_norm, axis=1, keepdims=True)
        # Map from [-1, 1] to [0, 1]
        similarity = (similarity + 1.0) / 2.0
    elif mode == 'l2':
        # L2 distance (convert to similarity)
        distance = tf.sqrt(tf.reduce_sum(
            tf.square(src_semantic - dst_semantic), axis=1, keepdims=True) + 1e-8)
        # Convert distance to similarity (exponential kernel)
        similarity = tf.exp(-distance)
    elif mode == 'kl':
        # KL divergence (for probability distributions)
        src_semantic = src_semantic + 1e-8  # Avoid log(0)
        dst_semantic = dst_semantic + 1e-8
        kl_div = tf.reduce_sum(
            src_semantic * tf.log(src_semantic / dst_semantic), 
            axis=1, keepdims=True)
        # Convert to similarity
        similarity = tf.exp(-kl_div)
    else:
        raise ValueError(f"Unknown similarity mode: {mode}")
    
    return similarity


class SemanticGraphNetAutoCenter(object):
    """
    Semantic-aware Graph Neural Network layer with auto-center mechanism.
    
    This layer enhances the original GraphNetAutoCenter by incorporating
    semantic consistency into edge feature calculations.
    """
    
    def __init__(self,
                 edge_feature_fn=multi_layer_neural_network_fn,
                 aggregation_fn=graph_scatter_max_fn,
                 update_fn=multi_layer_neural_network_fn,
                 auto_offset_fn=multi_layer_neural_network_fn,
                 semantic_mode='cosine'):
        self._edge_feature_fn = edge_feature_fn
        self._aggregation_fn = aggregation_fn
        self._update_fn = update_fn
        self._auto_offset_fn = auto_offset_fn
        self._semantic_mode = semantic_mode
    
    def apply_regular(self,
                     input_vertex_features,
                     input_vertex_coordinates,
                     input_vertex_semantics,
                     NOT_USED,
                     edges,
                     edge_MLP_depth_list=None,
                     edge_MLP_normalization_type='fused_BN_center',
                     edge_MLP_activation_type='ReLU',
                     update_MLP_depth_list=None,
                     update_MLP_normalization_type='fused_BN_center',
                     update_MLP_activation_type='ReLU',
                     auto_offset=False,
                     auto_offset_MLP_depth_list=None,
                     auto_offset_MLP_normalization_type='fused_BN_center',
                     auto_offset_MLP_feature_activation_type='ReLU',
                     use_semantic_consistency=True,
                     semantic_weight=1.0):
        """
        Apply semantic-aware graph network layer.
        
        Args:
            input_vertex_features: [N, M] geometric features
            input_vertex_coordinates: [N, 3] vertex coordinates
            input_vertex_semantics: [N, C] semantic features
            NOT_USED: for API compatibility
            edges: [K, 2] edge list (src, dst)
            use_semantic_consistency: whether to use semantic edge weighting
            semantic_weight: weight for semantic consistency factor
            ... (other args same as original GraphNetAutoCenter)
            
        Returns:
            output_vertex_features: [N, M] updated vertex features
        """
        # Gather source and destination vertices
        s_vertex_features = tf.gather(input_vertex_features, edges[:, 0])
        s_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 0])
        s_vertex_semantics = tf.gather(input_vertex_semantics, edges[:, 0])
        
        # Compute coordinate offset (if enabled)
        if auto_offset:
            offset = self._auto_offset_fn(input_vertex_features,
                                          Ks=auto_offset_MLP_depth_list,
                                          is_logits=True,
                                          normalization_type=auto_offset_MLP_normalization_type,
                                          activation_type=auto_offset_MLP_feature_activation_type)
            input_vertex_coordinates = input_vertex_coordinates + offset
        
        # Gather destination vertices
        d_vertex_coordinates = tf.gather(input_vertex_coordinates, edges[:, 1])
        d_vertex_semantics = tf.gather(input_vertex_semantics, edges[:, 1])
        
        # Prepare edge features with semantic information
        edge_features = tf.concat(
            [s_vertex_features, 
             s_vertex_coordinates - d_vertex_coordinates,
             s_vertex_semantics],
            axis=-1)
        
        with tf.variable_scope('extract_edge_features'):
            # Extract edge features
            edge_features = self._edge_feature_fn(
                edge_features,
                Ks=edge_MLP_depth_list,
                is_logits=False,
                normalization_type=edge_MLP_normalization_type,
                activation_type=edge_MLP_activation_type)
            
            # Apply semantic consistency weighting
            if use_semantic_consistency:
                semantic_similarity = compute_semantic_similarity(
                    s_vertex_semantics, d_vertex_semantics, 
                    mode=self._semantic_mode)
                # Weight edge features by semantic similarity
                edge_features = edge_features * (1.0 + semantic_weight * semantic_similarity)
            
            # Aggregate edge features
            aggregated_edge_features = self._aggregation_fn(
                edge_features,
                edges[:, 1],
                tf.shape(input_vertex_features)[0])
        
        # Update vertex features
        with tf.variable_scope('combined_features'):
            update_features = self._update_fn(
                aggregated_edge_features,
                Ks=update_MLP_depth_list,
                is_logits=True,
                normalization_type=update_MLP_normalization_type,
                activation_type=update_MLP_activation_type)
        
        # Residual connection
        output_vertex_features = update_features + input_vertex_features
        
        return output_vertex_features


class SemanticPointSetPooling(object):
    """
    Semantic-aware Point Set Pooling layer.
    
    Incorporates semantic information into local point set aggregation.
    """
    
    def __init__(self,
                 point_feature_fn=multi_layer_neural_network_fn,
                 aggregation_fn=graph_scatter_max_fn,
                 output_fn=multi_layer_neural_network_fn):
        self._point_feature_fn = point_feature_fn
        self._aggregation_fn = aggregation_fn
        self._output_fn = output_fn
    
    def apply_regular(self,
                     point_features,
                     point_coordinates,
                     point_semantics,
                     keypoint_indices,
                     set_indices,
                     point_MLP_depth_list=None,
                     point_MLP_normalization_type='fused_BN_center',
                     point_MLP_activation_type='ReLU',
                     output_MLP_depth_list=None,
                     output_MLP_normalization_type='fused_BN_center',
                     output_MLP_activation_type='ReLU'):
        """
        Apply semantic-aware point set pooling.
        
        Args:
            point_features: [N, M] point features
            point_coordinates: [N, 3] point coordinates
            point_semantics: [N, C] semantic features
            keypoint_indices: [K, 1] keypoint indices
            set_indices: [S, 2] (point_idx, set_idx) pairs
            
        Returns:
            set_features: [K, output_depth] pooled set features
        """
        # Gather points in each set
        point_set_features = tf.gather(point_features, set_indices[:, 0])
        point_set_coordinates = tf.gather(point_coordinates, set_indices[:, 0])
        point_set_semantics = tf.gather(point_semantics, set_indices[:, 0])
        
        # Gather keypoints
        point_set_keypoint_indices = tf.gather(keypoint_indices, set_indices[:, 1])
        point_set_keypoint_coordinates = tf.gather(
            point_coordinates, point_set_keypoint_indices[:, 0])
        
        # Relative coordinates
        point_set_coordinates = point_set_coordinates - point_set_keypoint_coordinates
        
        # Concatenate all features including semantics
        point_set_features = tf.concat(
            [point_set_features, point_set_coordinates, point_set_semantics],
            axis=-1)
        
        with tf.variable_scope('extract_vertex_features'):
            # Extract point features
            extracted_point_features = self._point_feature_fn(
                point_set_features,
                Ks=point_MLP_depth_list,
                is_logits=False,
                normalization_type=point_MLP_normalization_type,
                activation_type=point_MLP_activation_type)
            
            # Aggregate within each set
            set_features = self._aggregation_fn(
                extracted_point_features,
                set_indices[:, 1],
                tf.shape(keypoint_indices)[0])
        
        with tf.variable_scope('combined_features'):
            set_features = self._output_fn(
                set_features,
                Ks=output_MLP_depth_list,
                is_logits=False,
                normalization_type=output_MLP_normalization_type,
                activation_type=output_MLP_activation_type)
        
        return set_features


def create_semantic_layer_registry():
    """Create registry of semantic-aware GNN layers."""
    from functools import partial
    from models.gnn import multi_layer_fc_fn
    
    return {
        'semantic_graph_auto_center_net': SemanticGraphNetAutoCenter(
            edge_feature_fn=multi_layer_neural_network_fn,
            aggregation_fn=graph_scatter_max_fn,
            update_fn=multi_layer_neural_network_fn,
            auto_offset_fn=multi_layer_neural_network_fn,
            semantic_mode='cosine'
        ),
        'semantic_point_set_pooling': SemanticPointSetPooling(
            point_feature_fn=multi_layer_neural_network_fn,
            aggregation_fn=graph_scatter_max_fn,
            output_fn=multi_layer_neural_network_fn
        ),
    }

