#!/usr/bin/python2
# -*- coding: utf-8 -*-

"""
Training Dataset Visualization
"""

import argparse
# import glob
import os
import numpy as np
from PIL import Image

import rospy
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
# from segment_node import _make_point_field

def _make_point_field(num_field):
    msg_pf1 = pc2.PointField()
    msg_pf1.name = np.str('x')
    msg_pf1.offset = np.uint32(0)
    msg_pf1.datatype = np.uint8(7)
    msg_pf1.count = np.uint32(1)

    msg_pf2 = pc2.PointField()
    msg_pf2.name = np.str('y')
    msg_pf2.offset = np.uint32(4)
    msg_pf2.datatype = np.uint8(7)
    msg_pf2.count = np.uint32(1)

    msg_pf3 = pc2.PointField()
    msg_pf3.name = np.str('z')
    msg_pf3.offset = np.uint32(8)
    msg_pf3.datatype = np.uint8(7)
    msg_pf3.count = np.uint32(1)

    msg_pf4 = pc2.PointField()
    msg_pf4.name = np.str('intensity')
    msg_pf4.offset = np.uint32(16)
    msg_pf4.datatype = np.uint8(7)
    msg_pf4.count = np.uint32(1)

    if num_field == 4:
        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]

    msg_pf5 = pc2.PointField()
    msg_pf5.name = np.str('label')
    msg_pf5.offset = np.uint32(20)
    msg_pf5.datatype = np.uint8(4)
    msg_pf5.count = np.uint32(1)

    return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]

class KITTINode(object):
    """
    A ros node to publish training set 2D spherical surface image
    """

    def __init__(self, dataset_path='./data/lidar_2d',
                 pub_rate=10,
                 # pub_velody_points_topic='/kitti/velodyne_points',
                 pub_velody_points_topic='/switchedCloud',
                 pub_feature_map_topic='/squeeze_seg/feature_map',
                 pub_label_map_topic='/squeeze_seg/label_map'):
        """
        ros node spin in init function

        :param dataset_path:
        :param pub_feature_map_topic:
        :param pub_label_map_topic:
        :param pub_rate:
        """

        self._path = dataset_path + "/"
        self._pub_rate = pub_rate
        # publisher
        self._velodyne_points_pub = rospy.Publisher(pub_velody_points_topic, PointCloud2, queue_size=1)
        # self._feature_map_pub = rospy.Publisher(pub_feature_map_topic, ImageMsg, queue_size=1)
        # self._label_map_pub = rospy.Publisher(pub_label_map_topic, ImageMsg, queue_size=1)
        # ros node init
        rospy.init_node('npy_node', anonymous=True)
        rospy.loginfo("npy_node started.")
        # rospy.loginfo("publishing dataset %s in '%s'+'%s' topic with %d(hz)...", self._path,
        #               pub_feature_map_topic, pub_label_map_topic, self._pub_rate)

        header = Header()
        header.stamp = rospy.Time()
        header.frame_id = "velodyne"

        rate = rospy.Rate(self._pub_rate)
        cnt = 0

        bin_files = []
        if os.path.isdir(self._path):
            for f in os.listdir(self._path):
                if os.path.isdir(f):
                    continue
                else:
                    bin_files.append(f)
        bin_files.sort()

        # for f in glob.iglob(self.path_):
        for f in bin_files:
            if rospy.is_shutdown():
                break

            self.publish_pointcloud(self._path + "/" + f, header)
            cnt += 1

            rate.sleep()

        rospy.logwarn("%d frames published.", cnt)

    def publish_pointcloud(self, bin_file, header):
        # record = np.load(bin_file).astype(np.float32, copy=False)
        record = np.fromfile(bin_file, dtype=np.float32).reshape(-1,4) #(Nx4) xyzi
        lidar = record    # x, y, z, intensity
        # print lidar
        x = lidar[:, 0].reshape(-1)
        y = lidar[:, 1].reshape(-1)
        z = lidar[:, 2].reshape(-1)
        i = lidar[:, 3].reshape(-1)
        # label = lidar[:, :, -1].reshape(-1)

        # cloud = np.stack((x, y, z, i, label))
        cloud = np.stack((x, y, z, i))

        # label = record[:, :, 5]     # point-wise label
        # label = _normalize(label)
        # g=p*R+q*G+t*B, where p=0.2989,q=0.5870,t=0.1140
        # p = 0.2989;q = 0.5870;t = 0.1140
        # label_3d = np.dstack((p*label, q*label, t*label))
        # label_3d = np.zeros((label.shape[0], label.shape[1], 3))
        # label_3d[np.where(label==0)] = [1., 1., 1.]
        # label_3d[np.where(label==1)] = [0., 1., 0.]
        # label_3d[np.where(label==2)] = [1., 1., 0.]
        # label_3d[np.where(label==3)] = [0., 1., 1.]
        # print label_3d
        # print np.min(label)
        # print np.max(label)

        # insert label into lidar infos
        # lidar[np.where(label==1)] = [0., 1., 0., 0., 0.]
        # lidar[np.where(label==2)] = [1., 1., 0., 0., 0.]
        # lidar[np.where(label==3)] = [0., 1., 1., 0., 0.]
        # generated feature map from LiDAR data
        ##x/y/z
        # feature_map = Image.fromarray(
        #     (255 * _normalize(lidar[:, :, 0])).astype(np.uint8))
        ##depth map
        # feature_map = Image.fromarray(
        #     (255 * _normalize(lidar[:, :, 4])).astype(np.uint8))
        ##intensity map
        # feature_map = Image.fromarray(
        #     (255 * _normalize(lidar[:, :, 3])).astype(np.uint8))
        # feature_map = Image.fromarray(
        #     (255 * _normalize(lidar[:, :, :3])).astype(np.uint8))
        # generated label map from LiDAR data
        # label_map = Image.fromarray(
        #     (255 * _normalize(label_3d)).astype(np.uint8))

        # msg_points = pc2.create_cloud(header=header,
        #                                fields=_make_point_field(cloud.shape[0]),
        #                                points=cloud.T)
        msg_points = pc2.create_cloud(header=header, fields=_make_point_field(cloud.shape[0]), points=cloud.T)
        # msg_points.header = header
        # msg_feature = ImageConverter.to_ros(feature_map)
        # msg_feature.header = header
        # msg_label = ImageConverter.to_ros(label_map)
        # msg_label.header = header

        self._velodyne_points_pub.publish(msg_points)
        # self._feature_map_pub.publish(msg_feature)
        # self._label_map_pub.publish(msg_label)

        file_name = bin_file.strip('.bin').split('/')[-1]
        rospy.loginfo("%s published.", file_name)

if __name__ == '__main__':
    # parse arguments from command line
    parser = argparse.ArgumentParser(description='LiDAR point cloud 2D spherical surface publisher')
    # parser.add_argument('--dataset_path', type=str,
    #                     help='the path of training dataset, default `/media/data/kitti/data_object_velodyne/training/velodyne`',
    #                     default='/media/data/kitti/data_object_velodyne/training/velodyne')#/media/data/kitti/data_odometry_velodyne/dataset/sequences/
    parser.add_argument('--dataset_path', type=str,
                        help='the path of training dataset, default `/media/data/kitti/data_odometry_velodyne/dataset/sequences/00/velodyne`',
                        default='/media/data/kitti/data_odometry_velodyne/dataset/sequences/00/velodyne')
    parser.add_argument('--pub_rate', type=int,
                        help='the frequency(hz) of image published, default `10`',
                        default=5)
    parser.add_argument('--pub_velodyne_points_topic', type=str,
                        help='the 3D point cloud message topic to be published, default `/velodyne_points`',
                        default='/velodyne_points') #kitti/velodyne_points
    parser.add_argument('--pub_feature_map_topic', type=str,
                        help='the 2D spherical surface image message topic to be published, default `/squeeze_seg/feature_map`',
                        default='/squeeze_seg/feature_map')
    parser.add_argument('--pub_label_map_topic', type=str,
                        help='the corresponding ground truth label image message topic to be published, default `/squeeze_seg/label_map`',
                        default='/squeeze_seg/label_map')
    args = parser.parse_args()

    # start training_set_node
    node = KITTINode(dataset_path=args.dataset_path,
                           pub_rate=args.pub_rate,
                           pub_velody_points_topic=args.pub_velodyne_points_topic,
                           pub_feature_map_topic=args.pub_feature_map_topic,
                           pub_label_map_topic=args.pub_label_map_topic)

    rospy.logwarn('finished.')
