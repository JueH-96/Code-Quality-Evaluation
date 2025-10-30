from typing import List

class SegmentTree:
	def __init__(self, n):
		self.n = n
		self.size = 1
		while self.size < n:
			self.size *= 2
		self.data = [-10**18] * (2 * self.size)
	
	def update(self, i, val):
		i += self.size
		if self.data[i] < val:
			self.data[i] = val
			i //= 2
			while i:
				self.data[i] = max(self.data[2*i], self.data[2*i+1])
				i //= 2
	
	def query(self, l, r):
		l += self.size
		r += self.size
		res = -10**18
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
	def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
		n = len(nums1)
		points = []
		for i in range(n):
			points.append((nums1[i], nums2[i], nums1[i] + nums2[i]))
		
		all_ys = set()
		for a, b, s in points:
			all_ys.add(b)
		for x, y in queries:
			all_ys.add(y)
		all_ys = sorted(all_ys)
		mapping = {y_val: idx for idx, y_val in enumerate(all_ys)}
		m = len(all_ys)
		
		points.sort(key=lambda x: x[0], reverse=True)
		
		sorted_queries = []
		for idx, (x, y) in enumerate(queries):
			sorted_queries.append((x, y, idx))
		sorted_queries.sort(key=lambda x: x[0], reverse=True)
		
		seg_tree = SegmentTree(m)
		ans = [-1] * len(queries)
		j = 0
		for x, y, idx in sorted_queries:
			while j < n and points[j][0] >= x:
				a_val, b_val, s_val = points[j]
				pos_b = mapping[b_val]
				seg_tree.update(pos_b, s_val)
				j += 1
				
			pos_y = mapping[y]
			res = seg_tree.query(pos_y, m-1)
			if res < 0:
				ans[idx] = -1
			else:
				ans[idx] = res
				
		return ans