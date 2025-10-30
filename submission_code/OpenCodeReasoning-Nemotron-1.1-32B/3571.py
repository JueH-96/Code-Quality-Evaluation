class SegmentTree:
	def __init__(self, n):
		self.n = n
		self.size = 1
		while self.size < n:
			self.size *= 2
		self.data = [0] * (2 * self.size)
	
	def update(self, idx, value):
		idx += self.size
		if value > self.data[idx]:
			self.data[idx] = value
			while idx > 1:
				idx //= 2
				self.data[idx] = max(self.data[2*idx], self.data[2*idx+1])
	
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
		
		ys = sorted(set(y for x, y in coordinates))
		comp = {y: i+1 for i, y in enumerate(ys)}
		m = len(ys)
		
		points = []
		for i, (x, y) in enumerate(coordinates):
			points.append((x, y, i))
		
		points_forward = sorted(points, key=lambda p: (p[0], p[1]))
		seg_forward = SegmentTree(m)
		dp1 = [0] * n
		
		i = 0
		while i < n:
			j = i
			group = []
			curr_x = points_forward[i][0]
			while j < n and points_forward[j][0] == curr_x:
				x, y, idx = points_forward[j]
				comp_y = comp[y]
				r0 = comp_y - 2
				if r0 < 0:
					max_val = 0
				else:
					max_val = seg_forward.query(0, r0)
				dp1[idx] = max_val + 1
				group.append((x, y, idx))
				j += 1
			
			for (x, y, idx) in group:
				comp_y = comp[y]
				pos = comp_y - 1
				seg_forward.update(pos, dp1[idx])
			
			i = j
		
		points_backward = sorted(points, key=lambda p: (-p[0], -p[1]))
		seg_backward = SegmentTree(m)
		dp2 = [0] * n
		
		i = 0
		while i < n:
			j = i
			group = []
			curr_x = points_backward[i][0]
			while j < n and points_backward[j][0] == curr_x:
				x, y, idx = points_backward[j]
				comp_y = comp[y]
				l0 = comp_y
				r0 = m - 1
				if l0 > r0:
					max_val = 0
				else:
					max_val = seg_backward.query(l0, r0)
				dp2[idx] = max_val + 1
				group.append((x, y, idx))
				j += 1
			
			for (x, y, idx) in group:
				comp_y = comp[y]
				pos = comp_y - 1
				seg_backward.update(pos, dp2[idx])
			
			i = j
		
		result = dp1[k] + dp2[k] - 1
		return result