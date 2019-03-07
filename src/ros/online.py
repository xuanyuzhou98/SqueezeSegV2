#!/usr/bin/python2
# -*- coding: utf-8 -*-

"""
Segmentation on LiDAR point cloud using SqueezeSeg Neural Network
"""

import argparse
import tensorflow as tf
import rospy

from nodes.segment_node import SegmentNode

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'checkpoint', '../data/SqueezeSegV2/model.ckpt-30700',
    """Path to the model parameter file.""")
# tf.app.flags.DEFINE_string(
#     'checkpoint', '/home/kx/project/3D/SqueezeSeg-ROS/log_random_z/model.ckpt-274000',
#     """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/samples/*',
    """Input lidar scan to be detected. Can process glob input such as """
    """./data/samples/*.npy or single input.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/samples_out/', """Directory to dump output.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser(description='LiDAR point cloud semantic segmentation')
    parser.add_argument('--sub_topic', type=str,
                        help='the pointcloud message topic to be subscribed, default `/velodyne_points`',
                        default='/velodyne_points')
    # parser.add_argument('--sub_topic', type=str,
    #                     help='the pointcloud message topic to be subscribed, default `/kitti/velodyne_points`',
    #                     default='/kitti/velodyne_points')
    # parser.add_argument('--sub_topic', type=str,
    #                     help='the pointcloud message topic to be subscribed, default `/velodyne_points`',
    #                     default='/velodyne_points')
    parser.add_argument('--pub_topic', type=str,
                        help='the pointcloud message topic to be published, default `/squeeze_seg/points`',
                        default='/squeeze_seg/points')
    parser.add_argument('--pub_feature_map_topic', type=str,
                        help='the 2D spherical surface image message topic to be published, default `/squeeze_seg/feature_map`',
                        default='/squeeze_seg/feature_map')
    parser.add_argument('--pub_label_map_topic', type=str,
                        help='the corresponding ground truth label image message topic to be published, default `/squeeze_seg/label_map`',
                        default='/squeeze_seg/label_map')
    args = parser.parse_args()

    rospy.init_node('segment_node')
    node = SegmentNode(sub_topic=args.sub_topic,
                       pub_topic=args.pub_topic,
                       pub_feature_map_topic=args.pub_feature_map_topic,
                       pub_label_map_topic=args.pub_label_map_topic,
                       FLAGS=FLAGS)

    rospy.logwarn("finished.")