from typing import List

class Solution:
	def findMaximumLength(self, nums: List[int]) -> int:
		n = len(nums)
		prefix = [0] * (n + 1)
		for i in range(1, n + 1):
			prefix[i] = prefix[i - 1] + nums[i - 1]
		
		dp = [0] * n
		last = [0] * n
		
		MAX_RANGE = 2**35
		
		class Node:
			__slots__ = ['left', 'right', 'best']
			def __init__(self):
				self.left = None
				self.right = None
				self.best = (0, 0)
		
		class SegmentTree:
			def __init__(self):
				self.root = None
			
			def update(self, index, value):
				self.root = self._update(self.root, 0, MAX_RANGE, index, value)
			
			def _update(self, node, l, r, index, value):
				if node is None:
					node = Node()
				if l == r:
					if self.better(value, node.best) != node.best:
						node.best = value
					return node
				
				mid = (l + r) // 2
				if index <= mid:
					node.left = self._update(node.left, l, mid, index, value)
				else:
					node.right = self._update(node.right, mid + 1, r, index, value)
				
				left_best = node.left.best if node.left else (0, 0)
				right_best = node.right.best if node.right else (0, 0)
				node.best = self.better(left_best, right_best)
				return node
			
			def query(self, l, r):
				return self._query(self.root, 0, MAX_RANGE, l, r)
			
			def _query(self, node, seg_l, seg_r, l, r):
				if node is None or seg_r < l or seg_l > r:
					return (0, 0)
				if l <= seg_l and seg_r <= r:
					return node.best
				
				mid = (seg_l + seg_r) // 2
				left_res = (0, 0)
				right_res = (0, 0)
				if l <= mid:
					left_res = self._query(node.left, seg_l, mid, l, r)
				if r > mid:
					right_res = self._query(node.right, mid + 1, seg_r, l, r)
				
				return self.better(left_res, right_res)
			
			@staticmethod
			def better(a, b):
				if a[0] > b[0]:
					return a
				elif a[0] < b[0]:
					return b
				else:
					if a[1] > b[1]:
						return a
					else:
						return b
		
		tree = SegmentTree()
		
		for i in range(n):
			T = prefix[i + 1]
			candidate_dp0 = 1
			candidate_last0 = T
			
			candidate_data = tree.query(0, T)
			candidate_dp = 0
			candidate_last = None
			if candidate_data != (0, 0):
				k_val, p_val = candidate_data
				candidate_dp = k_val + 1
				candidate_last = T - p_val
			
			if candidate_dp > 0:
				dp[i] = candidate_dp
				last[i] = candidate_last
			else:
				dp[i] = candidate_dp0
				last[i] = candidate_last0
			
			key = prefix[i + 1] + last[i]
			tree.update(key, (dp[i], prefix[i + 1]))
		
		return dp[-1]