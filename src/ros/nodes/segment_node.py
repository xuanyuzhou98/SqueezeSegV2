#!/usr/bin/python2
# -*- coding: utf-8 -*-

"""
Segmentation ROS node on LiDAR point cloud using SqueezeSeg Neural Network
"""

import sys
import os.path
import numpy as np
from PIL import Image

# lib_path = os.path.abspath(os.path.join('..'))
# print lib_path
# sys.path.append(lib_path)
import tensorflow as tf

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image as ImageMsg
from std_msgs.msg import Header

sys.path.append("..")
from config import *
from nets import SqueezeSeg
from utils.util import *
# from utils.clock import Clock

def _normalize(x):
    return (x - x.min()) / (x.max() - x.min())

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

class ImageConverter(object):
    """
    Convert images/compressedimages to and from ROS

    From: https://github.com/CURG-archive/ros_rsvp
    """

    _ENCODINGMAP_PY_TO_ROS = {'L': 'mono8', 'RGB': 'rgb8',
                              'RGBA': 'rgba8', 'YCbCr': 'yuv422'}
    _ENCODINGMAP_ROS_TO_PY = {'mono8': 'L', 'rgb8': 'RGB',
                              'rgba8': 'RGBA', 'yuv422': 'YCbCr'}
    _PIL_MODE_CHANNELS = {'L': 1, 'RGB': 3, 'RGBA': 4, 'YCbCr': 3}

    @staticmethod
    def to_ros(img):
        """
        Convert a PIL/pygame image to a ROS compatible message (sensor_msgs.Image).
        """

        # Everything ok, convert PIL.Image to ROS and return it
        if img.mode == 'P':
            img = img.convert('RGB')

        rosimage = ImageMsg()
        rosimage.encoding = ImageConverter._ENCODINGMAP_PY_TO_ROS[img.mode]
        (rosimage.width, rosimage.height) = img.size
        rosimage.step = (ImageConverter._PIL_MODE_CHANNELS[img.mode]
                         * rosimage.width)
        rosimage.data = img.tobytes()
        return rosimage

class SegmentNode():
    """LiDAR point cloud segment ros node"""

    def __init__(self,
                 sub_topic, pub_topic, pub_feature_map_topic, pub_label_map_topic,
                 FLAGS):
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

        self._mc = kitti_squeezeSeg_config()
        self._mc.LOAD_PRETRAINED_MODEL = False
        # TODO(bichen): fix this hard-coded batch size.
        self._mc.BATCH_SIZE = 1 #1
        self._model = SqueezeSeg(self._mc)
        self._saver = tf.train.Saver(self._model.model_params)

        self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self._saver.restore(self._session, FLAGS.checkpoint)

        self._sub = rospy.Subscriber(sub_topic, PointCloud2, self.point_cloud_callback, queue_size=1)
        self._pub = rospy.Publisher(pub_topic, PointCloud2, queue_size=1)
        self._feature_map_pub = rospy.Publisher(pub_feature_map_topic, ImageMsg, queue_size=1)
        self._label_map_pub = rospy.Publisher(pub_label_map_topic, ImageMsg, queue_size=1)
        # self.score_threshold = rospy.get_param('~score_threshold', 0.1)
        # self.use_top_k = rospy.get_param('~use_top_k', 5)

        rospy.spin()

    def point_cloud_callback(self, cloud_msg):
        """

        :param cloud_msg:
        :return:
        """

        rospy.logwarn("subscribed. width: %d, height: %u, point_step: %d, row_step: %d",
                      cloud_msg.width, cloud_msg.height, cloud_msg.point_step, cloud_msg.row_step)

        pc = pc2.read_points(cloud_msg, skip_nans=False, field_names=("x", "y", "z","intensity"))
        # to conver pc into numpy.ndarray format
        np_p = np.array(list(pc))
        # perform fov filter by using hv_in_range
        cond = self.hv_in_range(x=np_p[:, 0],
                                y=np_p[:, 1],
                                z=np_p[:, 2],
                                fov=[-45, 45])
        # to rotate points according to calibrated points with velo2cam
        # np_p_ranged = np.stack((np_p[:,1],-np_p[:,2],np_p[:,0],np_p[:,3])).T
        np_p_ranged = np_p[cond]

        # get depth map
        lidar = self.pto_depth_map(velo_points=np_p_ranged, C=5)
        lidar_f = lidar.astype(np.float32)
        #normalize intensity from [0,255] to [0,1], as shown in KITTI dataset
        #dep_map[:,:,0] = (dep_map[:,:,0]-0)/np.max(dep_map[:,:,0])
        #dep_map = cv2.resize(src=dep_map,dsize=(512,64))

        # to perform prediction
        lidar_mask = np.reshape(
            (lidar[:, :, 4] > 0),
            [self._mc.ZENITH_LEVEL, self._mc.AZIMUTH_LEVEL, 1]
        )
        lidar_f = (lidar_f - self._mc.INPUT_MEAN) / self._mc.INPUT_STD
        lidar_f = np.append(lidar_f, lidar_mask, axis=2)
        pred_cls = self._session.run(
            self._model.pred_cls,
            feed_dict={
                self._model.lidar_input: [lidar_f],
                self._model.keep_prob: 1.0,
                self._model.lidar_mask: [lidar_mask]
            }
        )
        label = pred_cls[0]

        # # generated depth map from LiDAR data
        depth_map = Image.fromarray(
            (255 * _normalize(lidar[:, :, 3])).astype(np.uint8))

        label_3d = np.zeros((label.shape[0], label.shape[1], 3))
        label_3d[np.where(label==0)] = [1., 1., 1.]
        label_3d[np.where(label==1)] = [0., 1., 0.]
        label_3d[np.where(label==2)] = [1., 1., 0.]
        label_3d[np.where(label==3)] = [0., 1., 1.]

        ## point cloud for SqueezeSeg segments
        x = lidar[:, :, 0].reshape(-1)
        y = lidar[:, :, 1].reshape(-1)
        z = lidar[:, :, 2].reshape(-1)
        i = lidar[:, :, 3].reshape(-1)
        label = label.reshape(-1)
        # cond = (label!=0)
        # print(cond)
        for cls in range(4):
            x = np.append(x, 0)
            y = np.append(y, 0)
            z = np.append(z, 0)
            i = np.append(i, 0)
            label = np.append(label, cls)

        cloud = np.stack((x, y, z, i, label))
        # cloud = np.stack((x, y, z, i))

        label_map = Image.fromarray(
            (255 * _normalize(label_3d)).astype(np.uint8))

        header = Header()
        header.stamp = rospy.Time()
        header.frame_id = "velodyne"
        # feature map & label map
        msg_feature = ImageConverter.to_ros(depth_map)
        msg_feature.header = header
        msg_label = ImageConverter.to_ros(label_map)
        msg_label.header = header

        # point cloud segments
        # 4 PointFields as channel description
        msg_segment = pc2.create_cloud(header=header,
                                       fields=_make_point_field(cloud.shape[0]),
                                       points=cloud.T)

        self._feature_map_pub.publish(msg_feature)
        self._label_map_pub.publish(msg_label)
        self._pub.publish(msg_segment)
        # rospy.loginfo("Point cloud processed. Took %.6f ms.", clock.takeRealTime())

    def hv_in_range(self, x, y, z, fov, fov_type='h'):
        """
        Extract filtered in-range velodyne coordinates based on azimuth & elevation angle limit
        Args:
        `x`:velodyne points x array
        `y`:velodyne points y array
        `z`:velodyne points z array
        `fov`:a two element list, e.g.[-45,45]
        `fov_type`:the fov type, could be `h` or 'v',defualt in `h`
        Return:
        `cond`:condition of points within fov or not
        Raise:
        `NameError`:"fov type must be set between 'h' and 'v' "
        """
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        if fov_type == 'h':
            return np.logical_and(np.arctan2(y, x) > (-fov[1] * np.pi / 180), \
                                  np.arctan2(y, x) < (-fov[0] * np.pi / 180))
        elif fov_type == 'v':
            return np.logical_and(np.arctan2(z, d) < (fov[1] * np.pi / 180), \
                                  np.arctan2(z, d) > (fov[0] * np.pi / 180))
        else:
            raise NameError("fov type must be set between 'h' and 'v' ")

    def pto_depth_map(self, velo_points,
                      H=64, W=512, C=5, dtheta=np.radians(0.4), dphi=np.radians(90./512.0)):
        """
        Project velodyne points into front view depth map.
        :param velo_points: velodyne points in shape [:,4]
        :param H: the row num of depth map, could be 64(default), 32, 16
        :param W: the col num of depth map
        :param C: the channel size of depth map
            3 cartesian coordinates (x; y; z),
            an intensity measurement and
            range r = sqrt(x^2 + y^2 + z^2)
        :param dtheta: the delta theta of H, in radian
        :param dphi: the delta phi of W, in radian
        :return: `depth_map`: the projected depth map of shape[H,W,C]
        """

        x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]
        d = np.sqrt(x ** 2 + y ** 2 + z**2)
        r = np.sqrt(x ** 2 + y ** 2)
        d[d==0] = 0.000001
        r[r==0] = 0.000001
        phi = np.radians(45.) - np.arcsin(y/r)
        phi_ = (phi/dphi).astype(int)
        phi_[phi_<0] = 0
        phi_[phi_>=512] = 511

        # print(np.min(phi_))
        # print(np.max(phi_))
        #
        # print z
        # print np.radians(2.)
        # print np.arcsin(z/d)
        theta = np.radians(2.) - np.arcsin(z/d)
        # print theta
        theta_ = (theta/dtheta).astype(int)
        # print theta_
        theta_[theta_<0] = 0
        theta_[theta_>=64] = 63
        #print theta,phi,theta_.shape,phi_.shape
        # print(np.min((phi/dphi)),np.max((phi/dphi)))
        #np.savetxt('./dump/'+'phi'+"dump.txt",(phi_).astype(np.float32), fmt="%f")
        #np.savetxt('./dump/'+'phi_'+"dump.txt",(phi/dphi).astype(np.float32), fmt="%f")
        # print(np.min(theta_))
        # print(np.max(theta_))

        depth_map = np.zeros((H, W, C))
        # 5 channels according to paper
        if C == 5:
            depth_map[theta_, phi_, 0] = x
            depth_map[theta_, phi_, 1] = y
            depth_map[theta_, phi_, 2] = z
            depth_map[theta_, phi_, 3] = i
            depth_map[theta_, phi_, 4] = d
        else:
            depth_map[theta_, phi_, 0] = i
        return depth_map