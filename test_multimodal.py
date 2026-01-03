#!/usr/bin/env python3
"""测试多模态Point-GNN的所有新模块"""

import sys
import numpy as np

def test_imports():
    """测试所有模块导入"""
    print("=" * 60)
    print("测试1: 模块导入")
    print("=" * 60)
    
    try:
        from models.point_painting import PointPainter, SemanticSegmentationModel
        print("✅ point_painting 模块导入成功")
    except Exception as e:
        print(f"❌ point_painting 模块导入失败: {e}")
        return False
    
    try:
        from models.semantic_gnn import SemanticGraphNetAutoCenter, compute_semantic_similarity
        print("✅ semantic_gnn 模块导入成功")
    except Exception as e:
        print(f"❌ semantic_gnn 模块导入失败: {e}")
        return False
    
    try:
        from models.multimodal_models import MultiModalPointGNN
        print("✅ multimodal_models 模块导入成功")
    except Exception as e:
        print(f"❌ multimodal_models 模块导入失败: {e}")
        return False
    
    try:
        from models.enhanced_nms import nms_boxes_3d_semantic, adaptive_nms_threshold
        print("✅ enhanced_nms 模块导入成功")
    except Exception as e:
        print(f"❌ enhanced_nms 模块导入失败: {e}")
        return False
    
    try:
        from dataset.multimodal_kitti_dataset import MultiModalKittiDataset
        print("✅ multimodal_kitti_dataset 模块导入成功")
    except Exception as e:
        print(f"❌ multimodal_kitti_dataset 模块导入失败: {e}")
        return False
    
    print("\n所有模块导入成功！\n")
    return True


def test_point_painting():
    """测试PointPainting功能"""
    print("=" * 60)
    print("测试2: PointPainting功能")
    print("=" * 60)
    
    try:
        from models.point_painting import PointPainter, SemanticSegmentationModel
        
        # 创建PointPainter
        painter = PointPainter(num_classes=19)
        print("✅ PointPainter创建成功")
        
        # 创建语义分割模型（使用dummy模式）
        seg_model = SemanticSegmentationModel(model_path=None, num_classes=19)
        print("✅ SemanticSegmentationModel创建成功")
        
        # 测试dummy分割
        dummy_image = np.random.randint(0, 255, (375, 1242, 3), dtype=np.uint8)
        semantic_scores = seg_model._dummy_segmentation(dummy_image)
        print(f"✅ Dummy分割输出形状: {semantic_scores.shape}")
        print(f"   语义概率和: {semantic_scores[100, 100, :].sum():.3f} (应该接近1.0)")
        
        print("\nPointPainting功能测试通过！\n")
        return True
    except Exception as e:
        print(f"❌ PointPainting测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_semantic_gnn():
    """测试语义感知GNN"""
    print("=" * 60)
    print("测试3: 语义感知GNN")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        from models.semantic_gnn import compute_semantic_similarity
        
        # Disable eager execution for TF2.x compatibility
        try:
            tf.compat.v1.disable_eager_execution()
        except:
            pass
        
        # 创建测试数据
        src_semantic = tf.constant([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], dtype=tf.float32)
        dst_semantic = tf.constant([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]], dtype=tf.float32)
        
        # Use TF2.x compatible session
        try:
            Session = tf.compat.v1.Session
        except AttributeError:
            Session = tf.Session
        
        with Session() as sess:
            # 测试余弦相似度
            similarity = compute_semantic_similarity(src_semantic, dst_semantic, mode='cosine')
            sim_values = sess.run(similarity)
            print(f"✅ 余弦相似度计算成功: {sim_values.flatten()}")
            
            # 测试L2相似度
            similarity = compute_semantic_similarity(src_semantic, dst_semantic, mode='l2')
            sim_values = sess.run(similarity)
            print(f"✅ L2相似度计算成功: {sim_values.flatten()}")
        
        print("\n语义GNN功能测试通过！\n")
        return True
    except Exception as e:
        print(f"❌ 语义GNN测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_enhanced_nms():
    """测试增强的NMS"""
    print("=" * 60)
    print("测试4: 增强的NMS")
    print("=" * 60)
    
    try:
        from models.enhanced_nms import adaptive_nms_threshold, rescore_boxes_with_semantics
        
        # 测试自适应阈值
        box_labels = np.array([1, 1, 3, 3, 5])  # Car, Car, Pedestrian, Pedestrian, Cyclist
        thresholds = adaptive_nms_threshold(box_labels)
        print(f"✅ 自适应阈值: {thresholds}")
        print(f"   Car阈值: {thresholds[0]}, Pedestrian阈值: {thresholds[2]}")
        
        # 测试语义重评分
        detection_scores = np.array([0.8, 0.7, 0.6, 0.5, 0.9])
        semantic_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.95])
        rescored = rescore_boxes_with_semantics(
            box_labels, None, detection_scores, semantic_scores, semantic_weight=0.3
        )
        print(f"✅ 语义重评分:")
        print(f"   原始分数: {detection_scores}")
        print(f"   重评分后: {rescored}")
        
        print("\n增强NMS功能测试通过！\n")
        return True
    except Exception as e:
        print(f"❌ 增强NMS测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """测试配置加载"""
    print("=" * 60)
    print("测试5: 配置文件")
    print("=" * 60)
    
    try:
        from util.config_util import load_config
        
        config_path = 'checkpoints/car_auto_T3_train/config'
        config = load_config(config_path)
        
        print(f"✅ 配置文件加载成功")
        print(f"   模型名称: {config['model_name']}")
        print(f"   类别数量: {config['num_classes']}")
        print(f"   标签方法: {config['label_method']}")
        print(f"   输入特征: {config['input_features']}")
        
        print("\n配置文件测试通过！\n")
        return True
    except Exception as e:
        print(f"❌ 配置文件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("多模态Point-GNN功能测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 运行所有测试
    results.append(("模块导入", test_imports()))
    results.append(("PointPainting", test_point_painting()))
    results.append(("语义GNN", test_semantic_gnn()))
    results.append(("增强NMS", test_enhanced_nms()))
    results.append(("配置文件", test_config()))
    
    # 打印总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:20s}: {status}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！多模态Point-GNN已准备就绪。")
        print("\n下一步:")
        print("  1. 准备KITTI数据集")
        print("  2. 运行推理: python3 run_multimodal.py checkpoints/car_auto_T3_train/ --dataset_root_dir PATH")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查错误信息。")
        return 1


if __name__ == '__main__':
    sys.exit(main())

