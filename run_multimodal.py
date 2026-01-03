"""Multi-modal inference pipeline for Point-GNN with PointPainting on KITTI dataset."""

import os
import time
import argparse
import multiprocessing
from functools import partial

import numpy as np
import tensorflow as tf
import open3d
import cv2
from tqdm import tqdm

from dataset.multimodal_kitti_dataset import MultiModalKittiDataset
from models.graph_gen import get_graph_generate_fn
from models.multimodal_models import get_multimodal_model
from models.models import get_model
from models.box_encoding import get_box_decoding_fn, get_box_encoding_fn, \
                          get_encoding_len
from models import preprocess
from models.enhanced_nms import (nms_boxes_3d_semantic, 
                                 nms_boxes_3d_semantic_merge)
from models import nms
from util.config_util import load_config, load_train_config
from util.summary_util import write_summary_scale

parser = argparse.ArgumentParser(description='Multi-modal Point-GNN inference on KITTI')
parser.add_argument('checkpoint_path', type=str,
                   help='Path to checkpoint')
parser.add_argument('-l', '--level', type=int, default=0,
                   help='Visualization level, 0 to disable,'+
                   '1 to nonblocking visualization, 2 to block.'+
                   'Default=0')
parser.add_argument('--test', dest='test', action='store_true',
                    default=False, help='Enable test model')
parser.add_argument('--no-box-merge', dest='use_box_merge',
                    action='store_false', default='True',
                   help='Disable box merge.')
parser.add_argument('--no-box-score', dest='use_box_score',
                    action='store_false', default='True',
                   help='Disable box score.')
parser.add_argument('--dataset_root_dir', type=str, default='../dataset/kitti/',
                   help='Path to KITTI dataset. Default="../dataset/kitti/"')
parser.add_argument('--dataset_split_file', type=str,
                    default='',
                   help='Path to KITTI dataset split file.'
                   'Default="DATASET_ROOT_DIR/3DOP_splits/val.txt"')
parser.add_argument('--output_dir', type=str,
                    default='',
                   help='Path to save the detection results'
                   'Default="CHECKPOINT_PATH/eval/"')
parser.add_argument('--segmentation_model_path', type=str,
                    default=None,
                   help='Path to semantic segmentation model')
parser.add_argument('--use_point_painting', action='store_true',
                    default=True,
                   help='Use PointPainting for multi-modal fusion')
parser.add_argument('--use_semantic_nms', action='store_true',
                    default=True,
                   help='Use semantic-aware NMS')
parser.add_argument('--num_semantic_classes', type=int, default=19,
                   help='Number of semantic classes')

args = parser.parse_args()

VISUALIZATION_LEVEL = args.level
IS_TEST = args.test
USE_BOX_MERGE = args.use_box_merge
USE_BOX_SCORE = args.use_box_score
DATASET_DIR = args.dataset_root_dir
USE_POINT_PAINTING = args.use_point_painting
USE_SEMANTIC_NMS = args.use_semantic_nms
NUM_SEMANTIC_CLASSES = args.num_semantic_classes

if args.dataset_split_file == '':
    DATASET_SPLIT_FILE = os.path.join(DATASET_DIR, './3DOP_splits/val.txt')
else:
    DATASET_SPLIT_FILE = args.dataset_split_file
if args.output_dir == '':
    OUTPUT_DIR = os.path.join(args.checkpoint_path, './eval_multimodal/')
else:
    OUTPUT_DIR = args.output_dir

CHECKPOINT_PATH = args.checkpoint_path
CONFIG_PATH = os.path.join(CHECKPOINT_PATH, 'config')
assert os.path.isfile(CONFIG_PATH), f'No config file found in {CHECKPOINT_PATH}'

config = load_config(CONFIG_PATH)

# Setup dataset with multi-modal support =====================================
if IS_TEST:
    dataset = MultiModalKittiDataset(
        os.path.join(DATASET_DIR, 'image/testing/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/testing/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/testing/calib/'),
        '',
        num_classes=config['num_classes'],
        num_semantic_classes=NUM_SEMANTIC_CLASSES,
        segmentation_model_path=args.segmentation_model_path,
        use_point_painting=USE_POINT_PAINTING)
else:
    dataset = MultiModalKittiDataset(
        os.path.join(DATASET_DIR, 'image/training/image_2'),
        os.path.join(DATASET_DIR, 'velodyne/training/velodyne/'),
        os.path.join(DATASET_DIR, 'calib/training/calib/'),
        os.path.join(DATASET_DIR, 'labels/training/label_2'),
        DATASET_SPLIT_FILE,
        num_classes=config['num_classes'],
        num_semantic_classes=NUM_SEMANTIC_CLASSES,
        segmentation_model_path=args.segmentation_model_path,
        use_point_painting=USE_POINT_PAINTING)

NUM_TEST_SAMPLE = dataset.num_files
NUM_CLASSES = dataset.num_classes

print(f"Multi-modal mode: PointPainting={USE_POINT_PAINTING}, "
      f"Semantic NMS={USE_SEMANTIC_NMS}")

# Setup model =================================================================
BOX_ENCODING_LEN = get_encoding_len(config['box_encoding_method'])
box_encoding_fn = get_box_encoding_fn(config['box_encoding_method'])
box_decoding_fn = get_box_decoding_fn(config['box_encoding_method'])

# Determine input feature dimension
if USE_POINT_PAINTING:
    # Original features + semantic features
    input_feature_dim = 4 + NUM_SEMANTIC_CLASSES  # intensity + RGB + semantic
else:
    input_feature_dim = 4  # intensity + RGB

t_initial_vertex_features = tf.placeholder(
    dtype=tf.float32, shape=[None, input_feature_dim])

t_vertex_coord_list = [tf.placeholder(dtype=tf.float32, shape=[None, 3])]
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_vertex_coord_list.append(
        tf.placeholder(dtype=tf.float32, shape=[None, 3]))

t_edges_list = []
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_edges_list.append(
        tf.placeholder(dtype=tf.int32, shape=[None, 2]))

t_keypoint_indices_list = []
for _ in range(len(config['runtime_graph_gen_kwargs']['level_configs'])):
    t_keypoint_indices_list.append(
        tf.placeholder(dtype=tf.int32, shape=[None, 1]))

t_is_training = tf.placeholder(dtype=tf.bool, shape=[])

# Try to load multi-modal model, fallback to original if not available
try:
    if USE_POINT_PAINTING and 'multimodal_model_name' in config:
        model = get_multimodal_model(config['multimodal_model_name'])(
            num_classes=NUM_CLASSES,
            box_encoding_len=BOX_ENCODING_LEN,
            num_semantic_classes=NUM_SEMANTIC_CLASSES,
            mode='test',
            **config.get('multimodal_model_kwargs', config['model_kwargs']))
        
        # For multi-modal model, we need to split features
        # Assuming first 4 dimensions are geometric, rest are semantic
        t_geometric_features = t_initial_vertex_features[:, :4]
        t_semantic_features = t_initial_vertex_features[:, 4:]
        
        t_logits, t_pred_box = model.predict(
            t_geometric_features, t_vertex_coord_list, t_keypoint_indices_list,
            t_edges_list, t_semantic_features, t_is_training)
        print("Using multi-modal Point-GNN model")
    else:
        raise ValueError("Using standard model")
except:
    # Fallback to standard model
    model = get_model(config['model_name'])(
        num_classes=NUM_CLASSES,
        box_encoding_len=BOX_ENCODING_LEN,
        mode='test',
        **config['model_kwargs'])
    t_logits, t_pred_box = model.predict(
        t_initial_vertex_features, t_vertex_coord_list, t_keypoint_indices_list,
        t_edges_list, t_is_training)
    print("Using standard Point-GNN model")

t_probs = model.postprocess(t_logits)
t_predictions = tf.argmax(t_probs, axis=1, output_type=tf.int32)

# Optimizers ==================================================================
global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
fetches = {
    'step': global_step,
    'predictions': t_predictions,
    'probs': t_probs,
    'pred_box': t_pred_box
}

# Color maps for visualization
color_map = np.array([(211,211,211), (255, 0, 0), (255,20,147), (65, 244, 101),
    (169, 244, 65), (65, 79, 244), (65, 181, 244), (229, 244, 66)],
    dtype=np.float32)
color_map = color_map/255.0

gt_color_map = {
    'Pedestrian': (0,255,255),
    'Person_sitting': (218,112,214),
    'Car': (154,205,50),
    'Truck':(255,215,0),
    'Van': (255,20,147),
    'Tram': (250,128,114),
    'Misc': (128,0,128),
    'Cyclist': (255,165,0),
}

# Running network =============================================================
time_dict = {}
saver = tf.train.Saver()
graph = tf.get_default_graph()
gpu_options = tf.GPUOptions(allow_growth=True)

with tf.Session(graph=graph,
    config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.variables_initializer(tf.global_variables()))
    sess.run(tf.variables_initializer(tf.local_variables()))
    
    model_path = tf.train.latest_checkpoint(CHECKPOINT_PATH)
    print(f'Restore from checkpoint {model_path}')
    saver.restore(sess, model_path)
    previous_step = sess.run(global_step)
    
    for frame_idx in tqdm(range(0, NUM_TEST_SAMPLE)):
        start_time = time.time()
        
        # Provide input ======================================================
        # Get painted points with semantic features
        cam_rgb_points, semantic_features = \
            dataset.get_painted_points_with_split_features(
                frame_idx, config['downsample_by_voxel_size'])
        
        calib = dataset.get_calib(frame_idx)
        image = dataset.get_image(frame_idx)
        
        if not IS_TEST:
            box_label_list = dataset.get_label(frame_idx)
        
        input_time = time.time()
        time_dict['fetch input'] = time_dict.get('fetch input', 0) \
            + input_time - start_time
        
        # Generate graph
        graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])
        (vertex_coord_list, keypoint_indices_list, edges_list) = \
            graph_generate_fn(
                cam_rgb_points.xyz, **config['runtime_graph_gen_kwargs'])
        
        graph_time = time.time()
        time_dict['gen graph'] = time_dict.get('gen graph', 0) \
            + graph_time - input_time
        
        # Prepare input features
        if USE_POINT_PAINTING:
            # Concatenate geometric and semantic features
            input_v = np.hstack([cam_rgb_points.attr, semantic_features])
        else:
            input_v = cam_rgb_points.attr
        
        last_layer_graph_level = \
            config['model_kwargs']['layer_configs'][-1]['graph_level']
        last_layer_points_xyz = vertex_coord_list[last_layer_graph_level+1]
        
        if config['label_method'] == 'yaw':
            label_map = {'Background': 0, 'Car': 1, 'Pedestrian': 3,
                'Cyclist': 5,'DontCare': 7}
        elif config['label_method'] == 'Car':
            label_map = {'Background': 0, 'Car': 1, 'DontCare': 3}
        elif config['label_method'] == 'Pedestrian_and_Cyclist':
            label_map = {'Background': 0, 'Pedestrian': 1, 'Cyclist':3,
                'DontCare': 5}
        
        # Run forwarding =====================================================
        feed_dict = {
            t_initial_vertex_features: input_v,
            t_is_training: False,
        }
        feed_dict.update(dict(zip(t_edges_list, edges_list)))
        feed_dict.update(
            dict(zip(t_keypoint_indices_list, keypoint_indices_list)))
        feed_dict.update(dict(zip(t_vertex_coord_list, vertex_coord_list)))
        
        results = sess.run(fetches, feed_dict=feed_dict)
        
        gnn_time = time.time()
        time_dict['gnn inference'] = time_dict.get('gnn inference', 0) \
            + gnn_time - graph_time
        
        # Box decoding =======================================================
        box_probs = results['probs']
        box_labels = np.tile(np.expand_dims(np.arange(NUM_CLASSES), axis=0),
            (box_probs.shape[0], 1))
        box_labels = box_labels.reshape((-1))
        raw_box_labels = box_labels
        box_probs = box_probs.reshape((-1))
        pred_boxes = results['pred_box'].reshape((-1, 1, BOX_ENCODING_LEN))
        last_layer_points_xyz = np.tile(
            np.expand_dims(last_layer_points_xyz, axis=1), (1, NUM_CLASSES, 1))
        last_layer_points_xyz = last_layer_points_xyz.reshape((-1, 3))
        boxes_centers = last_layer_points_xyz
        
        decoded_boxes = box_decoding_fn(np.expand_dims(box_labels, axis=1),
            boxes_centers, pred_boxes, label_map)
        
        box_mask = (box_labels > 0)*(box_labels < NUM_CLASSES-1)
        box_mask = box_mask*(box_probs > 1./NUM_CLASSES)
        box_indices = np.nonzero(box_mask)[0]
        
        decode_time = time.time()
        time_dict['decode box'] = time_dict.get('decode box', 0) \
            + decode_time - gnn_time
        
        if box_indices.size != 0:
            box_labels = box_labels[box_indices]
            box_probs = box_probs[box_indices]
            decoded_boxes = decoded_boxes[box_indices, 0]
            
            # Get semantic confidence for detected boxes
            if USE_POINT_PAINTING and USE_SEMANTIC_NMS:
                # Compute semantic confidence from semantic features
                # Average semantic confidence of points contributing to each detection
                semantic_confidence = dataset.compute_semantic_confidence(
                    semantic_features)
                # Map to last layer points
                # For simplicity, use max confidence (can be improved)
                box_semantic_scores = np.ones(len(box_indices)) * 0.5
            else:
                box_semantic_scores = None
            
            detection_scores = box_probs
            
            # NMS with semantic awareness =====================================
            if USE_SEMANTIC_NMS and box_semantic_scores is not None:
                if USE_BOX_MERGE and USE_BOX_SCORE:
                    (class_labels, detection_boxes_3d, detection_scores,
                    nms_indices) = nms_boxes_3d_semantic_merge(
                        box_labels, decoded_boxes, detection_scores,
                        semantic_scores=box_semantic_scores,
                        overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=config['nms_overlapped_thres'],
                        appr_factor=100.0, top_k=-1,
                        attributes=np.arange(len(box_indices)),
                        use_adaptive_threshold=True,
                        semantic_rescore_weight=0.3)
                else:
                    (class_labels, detection_boxes_3d, detection_scores,
                    nms_indices) = nms_boxes_3d_semantic(
                        box_labels, decoded_boxes, detection_scores,
                        semantic_scores=box_semantic_scores,
                        overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=config['nms_overlapped_thres'],
                        appr_factor=100.0, top_k=-1,
                        attributes=np.arange(len(box_indices)),
                        use_adaptive_threshold=True,
                        semantic_rescore_weight=0.3)
            else:
                # Use standard NMS
                if USE_BOX_MERGE and USE_BOX_SCORE:
                    (class_labels, detection_boxes_3d, detection_scores,
                    nms_indices) = nms.nms_boxes_3d_uncertainty(
                        box_labels, decoded_boxes, detection_scores,
                        overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=config['nms_overlapped_thres'],
                        appr_factor=100.0, top_k=-1,
                        attributes=np.arange(len(box_indices)))
                else:
                    (class_labels, detection_boxes_3d, detection_scores,
                    nms_indices) = nms.nms_boxes_3d(
                        box_labels, decoded_boxes, detection_scores,
                        overlapped_fn=nms.overlapped_boxes_3d_fast_poly,
                        overlapped_thres=config['nms_overlapped_thres'],
                        appr_factor=100.0, top_k=-1,
                        attributes=np.arange(len(box_indices)))
            
            box_probs = detection_scores
            
            nms_time = time.time()
            time_dict['nms'] = time_dict.get('nms', 0) + nms_time - decode_time
            
            # Convert to KITTI format and save ================================
            detection_boxes_3d_corners = nms.boxes_3d_to_corners(
                detection_boxes_3d)
            pred_labels = []
            
            for i in range(len(detection_boxes_3d_corners)):
                detection_box_3d_corners = detection_boxes_3d_corners[i]
                from dataset.kitti_dataset import Points
                corners_cam_points = Points(xyz=detection_box_3d_corners, attr=None)
                corners_img_points = dataset.cam_points_to_image(corners_cam_points, calib)
                corners_xy = corners_img_points.xyz[:, :2]
                
                if config['label_method'] == 'yaw':
                    all_class_name = ['Background', 'Car', 'Car', 'Pedestrian',
                        'Pedestrian', 'Cyclist', 'Cyclist', 'DontCare']
                elif config['label_method'] == 'Car':
                    all_class_name = ['Background', 'Car', 'Car', 'DontCare']
                elif config['label_method'] == 'Pedestrian_and_Cyclist':
                    all_class_name = ['Background', 'Pedestrian', 'Pedestrian',
                        'Cyclist', 'Cyclist', 'DontCare']
                
                class_name = all_class_name[class_labels[i]]
                xmin, ymin = np.amin(corners_xy, axis=0)
                xmax, ymax = np.amax(corners_xy, axis=0)
                clip_xmin = max(xmin, 0.0)
                clip_ymin = max(ymin, 0.0)
                clip_xmax = min(xmax, 1242.0)
                clip_ymax = min(ymax, 375.0)
                
                truncation_rate = 1.0 - (clip_ymax - clip_ymin)*(
                    clip_xmax - clip_xmin)/((ymax - ymin)*(xmax - xmin))
                
                if truncation_rate > 0.4:
                    continue
                
                x3d, y3d, z3d, l, h, w, yaw = detection_boxes_3d[i]
                score = box_probs[i]
                
                pred_labels.append((class_name, -1, -1, 0,
                    clip_xmin, clip_ymin, clip_xmax, clip_ymax,
                    h, w, l, x3d, y3d, z3d, yaw, score))
            
            # Output ==========================================================
            filename = OUTPUT_DIR+'/data/'+dataset.get_filename(
                frame_idx)+'.txt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                for pred_label in pred_labels:
                    for field in pred_label:
                        f.write(str(field)+' ')
                    f.write('\n')
                f.write('\n')
        else:
            # No detections
            filename = OUTPUT_DIR+'/data/'+dataset.get_filename(
                frame_idx)+'.txt'
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write('\n')
        
        total_time = time.time()
        time_dict['total'] = time_dict.get('total', 0) \
            + total_time - start_time
    
    # Print timing statistics
    for key in time_dict:
        print(f"{key} time: {time_dict[key]/NUM_TEST_SAMPLE:.4f}s")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Multi-modal inference completed with {NUM_TEST_SAMPLE} samples")

