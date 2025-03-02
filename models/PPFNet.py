import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_raw_features_sizes = {'xyz': 3, 'dxyz': 3, 'ppf': 4}
_raw_features_order = {'xyz': 0, 'dxyz': 1, 'ppf': 2}


def square_distance(src, dst):
	"""Calculate Euclid distance between each two points.
		src^T * dst = xn * xm + yn * ym + zn * zm；
		sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
		sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
		dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
			 = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

	Args:
		src: source points, [B, N, C]
		dst: target points, [B, M, C]
	Returns:
		dist: per-point square distance, [B, N, M]
	"""
	B, N, _ = src.shape
	_, M, _ = dst.shape
	dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
	dist += torch.sum(src ** 2, dim=-1)[:, :, None]
	dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
	return dist


def angle(v1: torch.Tensor, v2: torch.Tensor):
	"""Compute angle between 2 vectors

	For robustness, we use the same formulation as in PPFNet, i.e.
		angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
	This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0

	Args:
		v1: (B, *, 3)
		v2: (B, *, 3)

	Returns:

	"""

	cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
							  v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
							  v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
	cross_prod_norm = torch.norm(cross_prod, dim=-1)
	dot_prod = torch.sum(v1 * v2, dim=-1)

	return torch.atan2(cross_prod_norm, dot_prod)


def query_ball_point(radius, nsample, xyz, new_xyz, itself_indices=None):
	""" Grouping layer in PointNet++.

	Inputs:
		radius: local region radius
		nsample: max sample number in local region
		xyz: all points, (B, N, C)
		new_xyz: query points, (B, S, C)
		itself_indices (Optional): Indices of new_xyz into xyz (B, S).
		  Used to try and prevent grouping the point itself into the neighborhood.
		  If there is insufficient points in the neighborhood, or if left is none, the resulting cluster will
		  still contain the center point.
	Returns:
		group_idx: grouped points index, [B, S, nsample]
	"""
	device = xyz.device
	B, N, C = xyz.shape
	_, S, _ = new_xyz.shape
	group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # (B, S, N)
	sqrdists = square_distance(new_xyz, xyz)

	if itself_indices is not None:
		# Remove indices of the center points so that it will not be chosen
		batch_indices = torch.arange(B, dtype=torch.long).to(device)[:, None].repeat(1, S)  # (B, S)
		row_indices = torch.arange(S, dtype=torch.long).to(device)[None, :].repeat(B, 1)  # (B, S)
		group_idx[batch_indices, row_indices, itself_indices] = N

	group_idx[sqrdists > radius ** 2] = N
	group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
	if itself_indices is not None:
		group_first = itself_indices[:, :, None].repeat([1, 1, nsample])
	else:
		group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
	mask = group_idx == N
	group_idx[mask] = group_first[mask]
	return group_idx


def index_points(points, idx):
	"""Array indexing, i.e. retrieves relevant points based on indices

	Args:
		points: input points data_loader, [B, N, C]
		idx: sample index data_loader, [B, S]. S can be 2 dimensional
	Returns:
		new_points:, indexed points data_loader, [B, S, C]
	"""
	device = points.device
	B = points.shape[0]
	view_shape = list(idx.shape)
	view_shape[1:] = [1] * (len(view_shape) - 1)
	repeat_shape = list(idx.shape)
	repeat_shape[0] = 1
	batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
	new_points = points[batch_indices, idx, :]
	return new_points


def farthest_point_sample(xyz, npoint):
	"""Iterative farthest point sampling

	Args:
		xyz: pointcloud data_loader, [B, N, C]
		npoint: number of samples
	Returns:
		centroids: sampled pointcloud index, [B, npoint]
	"""
	device = xyz.device
	B, N, C = xyz.shape
	centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
	distance = torch.ones(B, N).to(device) * 1e10
	farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
	batch_indices = torch.arange(B, dtype=torch.long).to(device)
	for i in range(npoint):
		centroids[:, i] = farthest
		centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
		dist = torch.sum((xyz - centroid) ** 2, -1)
		mask = dist < distance
		distance[mask] = dist[mask]
		farthest = torch.max(distance, -1)[1]
	return centroids


def sample_and_group_multi(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, normals: torch.Tensor,
						   returnfps: bool = False):
	"""Sample and group for xyz, dxyz and ppf features

	Args:
		npoint(int): Number of clusters (equivalently, keypoints) to sample.
					 Set to negative to compute for all points
		radius(int): Radius of cluster for computing local features
		nsample: Maximum number of points to consider per cluster
		xyz: XYZ coordinates of the points
		normals: Corresponding normals for the points (required for ppf computation)
		returnfps: Whether to return indices of FPS points and their neighborhood

	Returns:
		Dictionary containing the following fields ['xyz', 'dxyz', 'ppf'].
		If returnfps is True, also returns: grouped_xyz, fps_idx
	"""

	B, N, C = xyz.shape

	if npoint > 0:
		S = npoint
		fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
		new_xyz = index_points(xyz, fps_idx)
		nr = index_points(normals, fps_idx)[:, :, None, :]
	else:
		S = xyz.shape[1]
		fps_idx = torch.arange(0, xyz.shape[1])[None, ...].repeat(xyz.shape[0], 1).to(xyz.device)
		new_xyz = xyz
		nr = normals[:, :, None, :]

	idx = query_ball_point(radius, nsample, xyz, new_xyz, fps_idx)  # (B, npoint, nsample)
	grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, C)
	d = grouped_xyz - new_xyz.view(B, S, 1, C)  # d = p_r - p_i  (B, npoint, nsample, 3)
	ni = index_points(normals, idx)

	nr_d = angle(nr, d)
	ni_d = angle(ni, d)
	nr_ni = angle(nr, ni)
	d_norm = torch.norm(d, dim=-1)

	xyz_feat = d  # (B, npoint, n_sample, 3)
	ppf_feat = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1)  # (B, npoint, n_sample, 4)

	if returnfps:
		return {'xyz': new_xyz, 'dxyz': xyz_feat, 'ppf': ppf_feat}, grouped_xyz, fps_idx
	else:
		return {'xyz': new_xyz, 'dxyz': xyz_feat, 'ppf': ppf_feat}


def get_prepool(in_dim, out_dim):
	"""Shared FC part in PointNet before max pooling"""
	net = nn.Sequential(
		nn.Conv2d(in_dim, out_dim // 2, 1),
		nn.GroupNorm(8, out_dim // 2),
		nn.ReLU(),
		nn.Conv2d(out_dim // 2, out_dim // 2, 1),
		nn.GroupNorm(8, out_dim // 2),
		nn.ReLU(),
		nn.Conv2d(out_dim // 2, out_dim, 1),
		nn.GroupNorm(8, out_dim),
		nn.ReLU(),
	)
	return net


def get_postpool(in_dim, out_dim):
	"""Linear layers in PointNet after max pooling

	Args:
		in_dim: Number of input channels
		out_dim: Number of output channels. Typically smaller than in_dim

	"""
	net = nn.Sequential(
		nn.Conv1d(in_dim, in_dim, 1),
		nn.GroupNorm(8, in_dim),
		nn.ReLU(),
		nn.Conv1d(in_dim, out_dim, 1),
		nn.GroupNorm(8, out_dim),
		nn.ReLU(),
		nn.Conv1d(out_dim, out_dim, 1),
	)

	return net


class PPFNet(nn.Module):
	"""Feature extraction Module that extracts hybrid features"""
	def __init__(self, n_classes, features=['ppf', 'dxyz', 'xyz'], emb_dims=1024, radius=0.3, num_neighbors=64):
		super().__init__()

		self._logger = logging.getLogger(self.__class__.__name__)
		self._logger.info('Using early fusion, feature dim = {}'.format(emb_dims))
		self.radius = radius
		self.n_sample = num_neighbors

		self.features = sorted(features, key=lambda f: _raw_features_order[f])
		self._logger.info('Feature extraction using features {}'.format(', '.join(self.features)))

		# Layers
		raw_dim = np.sum([_raw_features_sizes[f] for f in self.features])  # number of channels after concat
		self.prepool = get_prepool(raw_dim, emb_dims * 2)
		self.postpool = get_postpool(emb_dims * 2, emb_dims)

		self.fc1 = nn.Linear(1024, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, n_classes)
		self.dropout = nn.Dropout(p=0.3)
		self.bn1 = nn.BatchNorm1d(512)
		self.bn2 = nn.BatchNorm1d(256)
		self.relu = nn.ReLU()

	def forward(self, xyz):
		"""Forward pass of the feature extraction network

		Args:
			xyz: (B, 3, N)

		Returns:
			cluster features (B, N_classes)

		"""
		xyz = xyz.permute(0, 2, 1)
		normals = xyz

		features = sample_and_group_multi(-1, self.radius, self.n_sample, xyz, normals)
		features['xyz'] = features['xyz'][:, :, None, :]

		# Gate and concat
		concat = []
		for i in range(len(self.features)):
			f = self.features[i]
			expanded = (features[f]).expand(-1, -1, self.n_sample, -1)
			concat.append(expanded)
		fused_input_feat = torch.cat(concat, -1)

		# Prepool_FC, pool, postpool-FC
		new_feat = fused_input_feat.permute(0, 3, 2, 1)  # [B, 10, n_sample, N]
		new_feat = self.prepool(new_feat)

		pooled_feat = torch.max(new_feat, 2)[0]  # Max pooling (B, C, N)

		post_feat = self.postpool(pooled_feat)  # Post pooling dense layers
		cluster_feat = post_feat.permute(0, 2, 1)
		cluster_feat = cluster_feat / torch.norm(cluster_feat, dim=-1, keepdim=True)

		cluster_feat = cluster_feat.max(1)[0]

		x = F.relu(self.bn1(self.fc1(cluster_feat)))
		x = F.relu(self.bn2(self.dropout(self.fc2(x))))
		x = self.fc3(x)

		x = F.log_softmax(x, dim=1)
		return x


if __name__ == '__main__':
    atensor = torch.rand(5, 3, 1000).cuda()
    anet = PPFNet(10).cuda()
    ares = anet(atensor)

    print(ares.size())



    pass




