"""This file implements multi-modal models for 3D object detection.

The models integrate semantic information from images with geometric features
from point clouds using the PointPainting mechanism.
"""

from functools import partial
import tensorflow as tf
try:
    import tensorflow.contrib.slim as slim
except (ImportError, AttributeError):
    import tf_slim as slim

from models.loss import focal_loss_sigmoid, focal_loss_softmax
from models import gnn
from models.semantic_gnn import (SemanticGraphNetAutoCenter, 
                                 SemanticPointSetPooling,
                                 create_semantic_layer_registry)

regularizer_dict = {
    'l2': slim.l2_regularizer,
    'l1': slim.l1_regularizer,
    'l1_l2': slim.l1_l2_regularizer,
}


class MultiModalPointGNN(object):
    """
    Multi-modal Point-GNN model with PointPainting.
    
    This model extends the original Point-GNN by incorporating semantic features
    from 2D images into the 3D graph neural network processing.
    """
    
    def __init__(self, num_classes, box_encoding_len, num_semantic_classes=19,
                 regularizer_type=None, regularizer_kwargs=None, 
                 layer_configs=None, mode=None, 
                 use_semantic_consistency=True,
                 semantic_weight=1.0):
        """
        Args:
            num_classes: int, number of object detection classes
            box_encoding_len: int, length of encoded bounding box
            num_semantic_classes: int, number of semantic segmentation classes
            regularizer_type: string, regularization type
            regularizer_kwargs: dict, regularizer parameters
            layer_configs: list of layer configurations
            mode: string, 'train', 'eval', or 'test'
            use_semantic_consistency: bool, use semantic edge weighting
            semantic_weight: float, weight for semantic consistency
        """
        self.num_classes = num_classes
        self.box_encoding_len = box_encoding_len
        self.num_semantic_classes = num_semantic_classes
        self.use_semantic_consistency = use_semantic_consistency
        self.semantic_weight = semantic_weight
        
        if regularizer_type is None:
            self._regularizer = None
        else:
            self._regularizer = regularizer_dict[regularizer_type](
                **regularizer_kwargs)
        
        self._layer_configs = layer_configs
        
        # Build layer registry with both standard and semantic layers
        self._default_layers_type = {
            # Standard layers
            'scatter_max_point_set_pooling': gnn.PointSetPooling(
                point_feature_fn=gnn.multi_layer_neural_network_fn,
                aggregation_fn=gnn.graph_scatter_max_fn,
                output_fn=gnn.multi_layer_neural_network_fn
            ),
            'scatter_max_graph_auto_center_net': gnn.GraphNetAutoCenter(
                edge_feature_fn=gnn.multi_layer_neural_network_fn,
                aggregation_fn=gnn.graph_scatter_max_fn,
                update_fn=gnn.multi_layer_neural_network_fn,
                auto_offset_fn=gnn.multi_layer_neural_network_fn,
            ),
            # Semantic-aware layers
            'semantic_graph_auto_center_net': SemanticGraphNetAutoCenter(
                edge_feature_fn=gnn.multi_layer_neural_network_fn,
                aggregation_fn=gnn.graph_scatter_max_fn,
                update_fn=gnn.multi_layer_neural_network_fn,
                auto_offset_fn=gnn.multi_layer_neural_network_fn,
                semantic_mode='cosine'
            ),
            'semantic_point_set_pooling': SemanticPointSetPooling(
                point_feature_fn=gnn.multi_layer_neural_network_fn,
                aggregation_fn=gnn.graph_scatter_max_fn,
                output_fn=gnn.multi_layer_neural_network_fn
            ),
            # Predictors
            'classaware_predictor': gnn.ClassAwarePredictor(
                cls_fn=partial(gnn.multi_layer_fc_fn, Ks=(64,), num_layer=2),
                loc_fn=partial(gnn.multi_layer_fc_fn, Ks=(64, 64,), num_layer=3)
            ),
            'classaware_predictor_128': gnn.ClassAwarePredictor(
                cls_fn=partial(gnn.multi_layer_fc_fn, Ks=(128,), num_layer=2),
                loc_fn=partial(gnn.multi_layer_fc_fn, Ks=(128, 128), num_layer=3)
            ),
            'classaware_separated_predictor': gnn.ClassAwareSeparatedPredictor(
                cls_fn=partial(gnn.multi_layer_fc_fn, Ks=(64,), num_layer=2),
                loc_fn=partial(gnn.multi_layer_fc_fn, Ks=(64, 64,), num_layer=3)
            ),
        }
        
        assert mode in ['train', 'eval', 'test'], 'Unsupported mode'
        self._mode = mode
    
    def predict(self,
                t_initial_vertex_features,
                t_vertex_coord_list,
                t_keypoint_indices_list,
                t_edges_list,
                t_semantic_features,
                is_training):
        """
        Predict objects with multi-modal features.
        
        Args:
            t_initial_vertex_features: [N, M] initial geometric features
            t_vertex_coord_list: list of [Ni, 3] vertex coordinates
            t_keypoint_indices_list: list of [Nj, 1] keypoint indices
            t_edges_list: list of [Ki, 2] edge lists
            t_semantic_features: [N, C] semantic features from PointPainting
            is_training: bool, training mode flag
            
        Returns:
            logits: [N_output, num_classes] classification logits
            box_encodings: [N_output, num_classes, box_encoding_len] box predictions
        """
        with slim.arg_scope([slim.batch_norm], is_training=is_training), \
             slim.arg_scope([slim.fully_connected],
                           weights_regularizer=self._regularizer):
            
            # Fuse initial features with semantic features
            tfeatures = tf.concat([t_initial_vertex_features, t_semantic_features], 
                                 axis=-1)
            
            # Store semantic features at each level
            semantic_features_list = [t_semantic_features]
            
            # Process through GNN layers
            tfeatures_list = [tfeatures]
            
            for idx in range(len(self._layer_configs) - 1):
                layer_config = self._layer_configs[idx]
                layer_scope = layer_config['scope']
                layer_type = layer_config['type']
                layer_kwargs = layer_config['kwargs']
                graph_level = layer_config['graph_level']
                
                t_vertex_coordinates = t_vertex_coord_list[graph_level]
                t_keypoint_indices = t_keypoint_indices_list[graph_level]
                t_edges = t_edges_list[graph_level]
                
                with tf.variable_scope(layer_scope, reuse=tf.AUTO_REUSE):
                    flgn = self._default_layers_type[layer_type]
                    print(f'@ level {graph_level} Graph, Add layer: {layer_scope}, type: {layer_type}')
                    
                    # Check if this is a semantic-aware layer
                    is_semantic_layer = 'semantic' in layer_type
                    
                    if is_semantic_layer:
                        # Use semantic-aware layer
                        layer_kwargs_with_semantic = layer_kwargs.copy()
                        layer_kwargs_with_semantic['use_semantic_consistency'] = \
                            self.use_semantic_consistency
                        layer_kwargs_with_semantic['semantic_weight'] = \
                            self.semantic_weight
                        
                        if 'device' in layer_config:
                            with tf.device(layer_config['device']):
                                tfeatures = flgn.apply_regular(
                                    tfeatures,
                                    t_vertex_coordinates,
                                    semantic_features_list[-1],
                                    t_keypoint_indices,
                                    t_edges,
                                    **layer_kwargs_with_semantic)
                        else:
                            tfeatures = flgn.apply_regular(
                                tfeatures,
                                t_vertex_coordinates,
                                semantic_features_list[-1],
                                t_keypoint_indices,
                                t_edges,
                                **layer_kwargs_with_semantic)
                        
                        # Propagate semantic features to next level
                        # For pooling layers, use max pooling of semantic features
                        if 'pooling' in layer_type:
                            next_semantic = tf.unsorted_segment_max(
                                tf.gather(semantic_features_list[-1], t_edges[:, 0]),
                                t_edges[:, 1],
                                tf.shape(t_keypoint_indices)[0])
                        else:
                            next_semantic = semantic_features_list[-1]
                        semantic_features_list.append(next_semantic)
                    else:
                        # Use standard layer
                        if 'device' in layer_config:
                            with tf.device(layer_config['device']):
                                tfeatures = flgn.apply_regular(
                                    tfeatures,
                                    t_vertex_coordinates,
                                    t_keypoint_indices,
                                    t_edges,
                                    **layer_kwargs)
                        else:
                            tfeatures = flgn.apply_regular(
                                tfeatures,
                                t_vertex_coordinates,
                                t_keypoint_indices,
                                t_edges,
                                **layer_kwargs)
                        
                        # Keep semantic features unchanged
                        semantic_features_list.append(semantic_features_list[-1])
                    
                    tfeatures_list.append(tfeatures)
                    print('Feature Dim: ' + str(tfeatures.shape[-1]))
            
            # Final prediction
            predictor_config = self._layer_configs[-1]
            assert predictor_config['type'] in [
                'classaware_predictor', 
                'classaware_predictor_128',
                'classaware_separated_predictor'
            ]
            
            predictor = self._default_layers_type[predictor_config['type']]
            print('Final Feature Dim: ' + str(tfeatures.shape[-1]))
            
            with tf.variable_scope(predictor_config['scope'], reuse=tf.AUTO_REUSE):
                logits, box_encodings = predictor.apply_regular(
                    tfeatures,
                    num_classes=self.num_classes,
                    box_encoding_len=self.box_encoding_len,
                    **predictor_config['kwargs'])
                print(f"Prediction {self.num_classes} classes")
        
        return logits, box_encodings
    
    def postprocess(self, logits):
        """Output predictions."""
        prob = tf.nn.softmax(logits, axis=-1)
        return prob
    
    def loss(self, logits, labels, pred_box, gt_box, valid_box,
             cls_loss_type='focal_sigmoid', cls_loss_kwargs={},
             loc_loss_type='huber_loss', loc_loss_kwargs={},
             loc_loss_weight=1.0, cls_loss_weight=1.0):
        """
        Compute loss (same as original Point-GNN).
        
        See models.py MultiLayerFastLocalGraphModelV2.loss for details.
        """
        if isinstance(loc_loss_weight, dict):
            loc_loss_weight = loc_loss_weight[self._mode]
        if isinstance(cls_loss_weight, dict):
            cls_loss_weight = cls_loss_weight[self._mode]
        if isinstance(cls_loss_type, dict):
            cls_loss_type = cls_loss_type[self._mode]
            cls_loss_kwargs = cls_loss_kwargs[self._mode]
        if isinstance(loc_loss_type, dict):
            loc_loss_type = loc_loss_type[self._mode]
            loc_loss_kwargs = loc_loss_kwargs[self._mode]
        
        loss_dict = {}
        
        # Classification loss
        assert cls_loss_type in ['softmax', 'top_k_softmax',
                                 'focal_sigmoid', 'focal_softmax']
        if cls_loss_type == 'softmax':
            point_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.squeeze(labels, axis=1), logits=logits)
            num_endpoint = tf.shape(point_loss)[0]
        elif cls_loss_type == 'focal_sigmoid':
            point_loss = focal_loss_sigmoid(labels, logits, **cls_loss_kwargs)
            num_endpoint = tf.shape(point_loss)[0]
        elif cls_loss_type == 'focal_softmax':
            point_loss = focal_loss_softmax(labels, logits, **cls_loss_kwargs)
            num_endpoint = tf.shape(point_loss)[0]
        elif cls_loss_type == 'top_k_softmax':
            point_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.squeeze(labels, axis=1), logits=logits)
            num_endpoint = tf.shape(point_loss)[0]
            k = cls_loss_kwargs['k']
            top_k_cls_loss, _ = tf.math.top_k(point_loss, k=k, sorted=True)
            point_loss = top_k_cls_loss
        
        cls_loss = cls_loss_weight * tf.reduce_mean(point_loss)
        
        # Localization loss
        batch_idx = tf.range(tf.shape(pred_box)[0])
        batch_idx = tf.expand_dims(batch_idx, axis=1)
        batch_idx = tf.concat([batch_idx, labels], axis=1)
        pred_box = tf.gather_nd(pred_box, batch_idx)
        pred_box = tf.expand_dims(pred_box, axis=1)
        
        if loc_loss_type == 'huber_loss':
            all_loc_loss = loc_loss_weight * tf.losses.huber_loss(
                gt_box, pred_box, delta=1.0, weights=valid_box,
                reduction=tf.losses.Reduction.NONE)
            all_loc_loss = tf.squeeze(all_loc_loss, axis=1)
            
            if 'classwise_loc_loss_weight' in loc_loss_kwargs and \
               self._mode == 'train':
                classwise_loc_loss_weight = loc_loss_kwargs[
                    'classwise_loc_loss_weight']
                classwise_loc_loss_weight = tf.gather(
                    classwise_loc_loss_weight, labels)
                all_loc_loss = all_loc_loss * classwise_loc_loss_weight
            
            num_valid_endpoint = tf.reduce_sum(valid_box)
            mean_loc_loss = tf.reduce_mean(all_loc_loss, axis=1)
            loc_loss = tf.div_no_nan(tf.reduce_sum(mean_loc_loss),
                                     num_valid_endpoint)
            
            classwise_loc_loss = []
            for class_idx in range(self.num_classes):
                class_mask = tf.where(tf.equal(tf.squeeze(labels, axis=1),
                                              tf.constant(class_idx, tf.int32)))
                l = tf.reduce_sum(tf.gather(all_loc_loss, class_mask), axis=0)
                l = tf.squeeze(l, axis=0)
                is_nan_mask = tf.is_nan(l)
                l = tf.where(is_nan_mask, tf.zeros_like(l), l)
                classwise_loc_loss.append(l)
            loss_dict['classwise_loc_loss'] = classwise_loc_loss
        
        # Regularization loss
        with tf.control_dependencies([tf.assert_equal(tf.is_nan(loc_loss), False)]):
            reg_loss = tf.reduce_sum(tf.losses.get_regularization_losses())
        
        loss_dict.update({
            'cls_loss': cls_loss,
            'loc_loss': loc_loss,
            'reg_loss': reg_loss,
            'num_endpoint': num_endpoint,
            'num_valid_endpoint': num_valid_endpoint
        })
        
        return loss_dict


def get_multimodal_model(model_name):
    """Fetch a multi-modal model class."""
    model_map = {
        'multimodal_point_gnn': MultiModalPointGNN,
    }
    return model_map.get(model_name, None)

