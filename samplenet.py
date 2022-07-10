import tensorflow as tf
import numpy as np
from builtins import object

from encoders_decoders import encoder_with_convs_and_symmetry, decoder_with_fc_only
from tf_ops.grouping.tf_grouping import group_point, knn_point
from tf_ops.CD import tf_nndistance

def simple_projection_and_continued_fps(full_pc, gen_pc, idx):
    batch_size = np.size(full_pc, 0)
    k = np.size(gen_pc, 1)
    out_pc = np.zeros_like(gen_pc)
    out_pc_idx = np.zeros([batch_size, k], dtype=int)
    n_unique_points = np.zeros([batch_size, 1])
    for ii in range(0, batch_size):
        best_idx = idx[ii]
        best_idx = unique(best_idx)
        n_unique_points[ii] = np.size(best_idx, 0)
        out_pc[ii], out_pc_idx[ii] = fps_from_given_indices(full_pc[ii], k, best_idx)
    return out_pc, out_pc_idx, n_unique_points
def fps_from_given_indices(pts, K, given_idx):
    farthest_pts = np.zeros((K, 3))
    idx = np.zeros(K, dtype=int)
    t = np.size(given_idx)
    farthest_pts[0:t] = pts[given_idx]
    if t > 1:
        idx[0:t] = given_idx[0:t]
    else:
        idx[0] = given_idx

    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, t):
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))

    for i in range(t, K):
        idx[i] = np.argmax(distances)
        farthest_pts[i] = pts[idx[i]]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts, idx

def unique(arr):
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx)]
def calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)

class SoftProjection(object):
    def __init__(
        self, group_size, initial_temperature=1.0, is_temperature_trainable=True
    ):
        """Computes a soft nearest neighbor point cloud.
        Arguments:
            group_size: An integer, number of neighbors in nearest neighborhood.
            initial_temperature: A positive real number, initialization constant for temperature parameter.
            is_temperature_trainable: bool.
        Inputs:
            point_cloud: A `Tensor` of shape (batch_size, num_in_points, 3), original point cloud.
            query_cloud: A `Tensor` of shape (batch_size, num_out_points, 3), generated point cloud
        Outputs:
            projected_point_cloud: A `Tensor` of shape (batch_size, num_out_points, 3),
                the query_cloud projected onto its group_size nearest neighborhood,
                controlled by the learnable temperature parameter.
            weights: A `Tensor` of shape (batch_size, num_out_points, group_size, 1),
                the projection weights of the query_cloud onto its group_size nearest neighborhood
            dist: A `Tensor` of shape (batch_size, num_out_points, group_size, 1),
                the square distance of each query point from its neighbors divided by squared temperature parameter
        """

        self._group_size = group_size

        # create temperature variable
        self._temperature = tf.get_variable(
            "temperature",
            initializer=tf.constant(initial_temperature, dtype=tf.float32),
            trainable=is_temperature_trainable,
            dtype=tf.float32,
        )

        self._temperature_safe = tf.maximum(self._temperature, 1e-2)

        # sigma is exposed for loss calculation
        self.sigma = self._temperature_safe ** 2

    def __call__(self, point_cloud, query_cloud, hard=False):
        return self.project(point_cloud, query_cloud, hard)

    def _group_points(self, point_cloud, query_cloud):
        group_size = self._group_size
        _, num_out_points, _ = query_cloud.shape

        # find nearest group_size neighbours in point_cloud
        _, idx = knn_point(group_size, point_cloud, query_cloud)
        grouped_points = group_point(point_cloud, idx)
        return grouped_points

    def _get_distances(self, grouped_points, query_cloud):
        group_size = self._group_size

        # remove centers to get absolute distances
        deltas = grouped_points - tf.tile(
            tf.expand_dims(query_cloud, 2), [1, 1, group_size, 1]
        )
        dist = tf.reduce_sum(deltas ** 2, axis=3, keepdims=True) / self.sigma
        return dist

    def project(self, point_cloud, query_cloud, hard):
        grouped_points = self._group_points(
            point_cloud, query_cloud
        )  # (batch_size, num_out_points, group_size, 3)
        dist = self._get_distances(grouped_points, query_cloud)

        # pass through softmax to get weights
        weights = tf.nn.softmax(-dist, axis=2)
        if hard:
            # convert softmax weights to one_hot encoding
            weights = tf.one_hot(tf.argmax(weights, axis=2), depth=self._group_size)
            weights = tf.transpose(weights, perm=[0, 1, 3, 2])

        # get weighted average of grouped_points
        projected_point_cloud = tf.reduce_sum(
            grouped_points * weights, axis=2
        )  # (batch_size, num_out_points, 3)
        return projected_point_cloud, weights, self._temperature_safe

#lambda:0.0001
def get_project_loss(pj):
    result=pj.sigma
    return result
