# Author: Xuanyu Zhou (xuanyu_zhou@berkeley.edu), Bichen Wu (bichen@berkeley.edu) 10/27/2018

"""SqueezeSegV2 model"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import joblib
from utils import util
import numpy as np
import tensorflow as tf
from nn_skeleton import ModelSkeleton

class SqueezeSeg(ModelSkeleton):
  def __init__(self, mc, gpu_id_1=0, gpu_id_2=1, gpu_id_3=2):
    # Distribute the tensors to gpus vertically
    with tf.device('/gpu:{}'.format(gpu_id_1)):
      ModelSkeleton.__init__(self, mc)
      self._add_forward_graph_1()
    with tf.device('/gpu:{}'.format(gpu_id_2)):
      self._add_forward_graph_2()
    with tf.device('/gpu:{}'.format(gpu_id_3)):
      self._add_forward_graph_3()
      self._add_output_graph()
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()
      self._add_summary_ops()

  def _add_forward_graph_1(self):
    """NN architecture."""

    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      print('loding pretrained model')
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)
    self.conv1 = self._conv_bn_layer(
            self.lidar_input, 'conv1', 'bias', 'scale',
            filters=64, size=3, stride=2, padding='SAME', freeze=False,
            conv_with_bias=True, stddev=0.001)
    self.ca1 = self._context_aggregation_layer('se1', self.conv1)

    self.conv1_skip = self._conv_bn_layer(
            self.lidar_input, 'conv1_skip', 'bias', 'scale',
            filters=64, size=1, stride=1, padding='SAME', freeze=False,
            conv_with_bias=True, stddev=0.001)

    pool1 = self._pooling_layer(
        'pool1', self.ca1, size=3, stride=2, padding='SAME')
    fire2 = self._fire_layer(
        'fire2', pool1, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    ca2 = self._context_aggregation_layer('se2', fire2)
    fire3 = self._fire_layer(
        'fire3', ca2, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    self.ca3 = self._context_aggregation_layer('se3', fire3)
    pool3 = self._pooling_layer(
        'pool3', self.ca3, size=3, stride=2, padding='SAME')
  
    fire4 = self._fire_layer(
        'fire4', pool3, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    self.fire5 = self._fire_layer(
        'fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    pool5 = self._pooling_layer(
        'pool5', self.fire5, size=3, stride=2, padding='SAME')
  
    fire6 = self._fire_layer(
        'fire6', pool5, s1x1=48, e1x1=192, e3x3=192, freeze=False)
    fire7 = self._fire_layer(
        'fire7', fire6, s1x1=48, e1x1=192, e3x3=192, freeze=False)
    fire8 = self._fire_layer(
        'fire8', fire7, s1x1=64, e1x1=256, e3x3=256, freeze=False)
    self.fire9 = self._fire_layer(
        'fire9', fire8, s1x1=64, e1x1=256, e3x3=256, freeze=False)

  def _add_forward_graph_2(self):
    mc = self.mc
    # Deconvolation
    fire10 = self._fire_deconv(
        'fire_deconv10', self.fire9, s1x1=64, e1x1=128, e3x3=128, factors=[1, 2],
        stddev=0.1)
    fire10_fuse = tf.add(fire10, self.fire5, name='fure10_fuse')

    fire11 = self._fire_deconv(
        'fire_deconv11', fire10_fuse, s1x1=32, e1x1=64, e3x3=64, factors=[1, 2],
        stddev=0.1)
    fire11_fuse = tf.add(fire11, self.ca3, name='fire11_fuse')

    fire12 = self._fire_deconv(
        'fire_deconv12', fire11_fuse, s1x1=16, e1x1=32, e3x3=32, factors=[1, 2],
        stddev=0.1)
    self.fire12_fuse = tf.add(fire12, self.ca1, name='fire12_fuse')
  
  def _add_forward_graph_3(self):
    mc = self.mc
    fire13 = self._fire_deconv(
        'fire_deconv13', self.fire12_fuse, s1x1=16, e1x1=32, e3x3=32, factors=[1, 2],
        stddev=0.1)
    fire13_fuse = tf.add(fire13, self.conv1_skip, name='fire13_fuse')

    drop13 = tf.nn.dropout(fire13_fuse, self.keep_prob, name='drop13')
    conv14 = self._conv_layer(
        'conv14_prob', drop13, filters=mc.NUM_CLASS, size=3, stride=1,
        padding='SAME', relu=False, stddev=0.1, init=True)
    
    bilateral_filter_weights = self._bilateral_filter_layer(
        'bilateral_filter', self.lidar_input[:, :, :, :3], # x, y, z
        thetas=[mc.BILATERAL_THETA_A, mc.BILATERAL_THETA_R],
        sizes=[mc.LCN_HEIGHT, mc.LCN_WIDTH], stride=1)

    self.output_prob = self._recurrent_crf_layer(
        'recurrent_crf', conv14, bilateral_filter_weights, 
        sizes=[mc.LCN_HEIGHT, mc.LCN_WIDTH], num_iterations=mc.RCRF_ITER,
        padding='SAME'
    )

  def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, stddev=0.001,
      freeze=False):
    """Fire layer constructor.
    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """
    sq1x1 = self._conv_bn_layer(
            inputs, layer_name+'/squeeze1x1', 'bias', 'scale',
            filters=s1x1, size=1, stride=1, padding='SAME', freeze=freeze,
            conv_with_bias=True, stddev=stddev)
    ex1x1 = self._conv_bn_layer(
            sq1x1, layer_name+'/expand1x1', 'bias', 'scale',
            filters=e1x1, size=1, stride=1, padding='SAME', freeze=freeze,
            conv_with_bias=True, stddev=stddev)
    ex3x3 = self._conv_bn_layer(
            sq1x1, layer_name+'/expand3x3', 'bias', 'scale',
            filters=e3x3, size=3, stride=1, padding='SAME', freeze=freeze,
            conv_with_bias=True, stddev=stddev)
    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')

  def _fire_deconv(self, layer_name, inputs, s1x1, e1x1, e3x3, 
                   factors=[1, 2], freeze=False, stddev=0.001):
    """Fire deconvolution layer constructor.
    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      factors: spatial upsampling factors.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """
    assert len(factors) == 2,'factors should be an array of size 2'

    ksize_h = factors[0] * 2 - factors[0] % 2
    ksize_w = factors[1] * 2 - factors[1] % 2
    sq1x1 = self._conv_bn_layer(
            inputs, layer_name+'/squeeze1x1', 'bias', 'scale',
            filters=s1x1, size=1, stride=1, padding='SAME', freeze=freeze,
            conv_with_bias=True, stddev=stddev)
    deconv = self._deconv_layer(
        layer_name+'/deconv', sq1x1, filters=s1x1, size=[ksize_h, ksize_w],
        stride=factors, padding='SAME', init='bilinear')
    ex1x1 = self._conv_bn_layer(
            deconv, layer_name+'/expand1x1', 'bias', 'scale',
            filters=e1x1, size=1, stride=1, padding='SAME', freeze=freeze,
            conv_with_bias=True, stddev=stddev)
    ex3x3 = self._conv_bn_layer(
            deconv, layer_name+'/expand3x3', 'bias', 'scale',
            filters=e3x3, size=3, stride=1, padding='SAME', freeze=freeze,
            conv_with_bias=True, stddev=stddev)
    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')
    
