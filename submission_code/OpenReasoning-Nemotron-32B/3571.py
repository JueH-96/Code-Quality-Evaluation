import itertools
from typing import List

class SegmentTree:
	def __init__(self, n):
		self.n = n
		self.size = 1
		while self.size < n:
			self.size *= 2
		self.data = [0] * (2 * self.size)
	
	def update(self, index, value):
		i = index + self.size
		if value <= self.data[i]:
			return
		self.data[i] = value
		i //= 2
		while i:
			self.data[i] = max(self.data[2*i], self.data[2*i+1])
			i //= 2

	def query(self, l, r):
		if l > r:
			return 0
		l += self.size
		r += self.size
		res = 0
		while l <= r:
			if l % 2 == 1:
				res = max(res, self.data[l])
				l += 1
			if r % 2 == 0:
				res = max(res, self.data[r])
				r -= 1
			l //= 2
			r //= 2
		return res

class Solution:
	def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
		n = len(coordinates)
		if n == 0:
			return 0
		
		ys = [point[1] for point in coordinates]
		sorted_ys = sorted(set(ys))
		comp = {y_val: idx for idx, y_val in enumerate(sorted_ys)}
		m = len(sorted_ys)
		
		points = []
		for i, (x, y) in enumerate(coordinates):
			points.append((x, y, i))
		
		points_end = sorted(points, key=lambda p: (p[0], p[1]))
		groups_end = []
		for x, group in itertools.groupby(points_end, key=lambda p: p[0]):
			groups_end.append(list(group))
		
		seg_end = SegmentTree(m)
		dp_end = [0] * n
		
		for group in groups_end:
			updates = []
			for x, y, orig_idx in group:
				idx_y = comp[y]
				if idx_y == 0:
					q_val = 0
				else:
					q_val = seg_end.query(0, idx_y-1)
				dp_end[orig_idx] = q_val + 1
				updates.append((idx_y, dp_end[orig_idx]))
			for idx_y, val in updates:
				seg_end.update(idx_y, val)
		
		points_start = sorted(points, key=lambda p: (-p[0], -p[1]))
		groups_start = []
		for x, group in itertools.groupby(points_start, key=lambda p: p[0]):
			groups_start.append(list(group))
		
		seg_start = SegmentTree(m)
		dp_start = [0] * n
		
		for group in groups_start:
			updates = []
			for x, y, orig_idx in group:
				idx_y = comp[y]
				if idx_y == m-1:
					q_val = 0
				else:
					q_val = seg_start.query(idx_y+1, m-1)
				dp_start[orig_idx] = q_val + 1
				updates.append((idx_y, dp_start[orig_idx]))
			for idx_y, val in updates:
				seg_start.update(idx_y, val)
		
		return dp_end[k] + dp_start[k] - 1