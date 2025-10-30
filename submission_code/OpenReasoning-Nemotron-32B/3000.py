import bisect
from typing import List

INF = 10**18

class SegmentTree:
	def __init__(self, size, init, func):
		self.n = size
		self.init = init
		self.func = func
		self.size = 1
		while self.size < size:
			self.size *= 2
		self.data = [init] * (2 * self.size)
	
	def update(self, index, value):
		i = index + self.size
		self.data[i] = value
		i //= 2
		while i:
			self.data[i] = self.func(self.data[2*i], self.data[2*i+1])
			i //= 2
			
	def query(self, l, r):
		l += self.size
		r += self.size
		res = self.init
		while l <= r:
			if l % 2 == 1:
				res = self.func(res, self.data[l])
				l += 1
			if r % 2 == 0:
				res = self.func(res, self.data[r])
				r -= 1
			l //= 2
			r //= 2
		return res

class Solution:
	def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
		n = len(nums)
		if x == 0:
			nums_sorted = sorted(nums)
			min_diff = INF
			for i in range(1, n):
				diff = nums_sorted[i] - nums_sorted[i-1]
				if diff == 0:
					return 0
				if diff < min_diff:
					min_diff = diff
			return min_diff
		else:
			all_vals = sorted(set(nums))
			n_val = len(all_vals)
			seg_max = SegmentTree(n_val, init=-INF, func=max)
			seg_min = SegmentTree(n_val, init=INF, func=min)
			
			ans = INF
			for j in range(x, n):
				val_to_add = nums[j-x]
				idx_add = bisect.bisect_left(all_vals, val_to_add)
				seg_max.update(idx_add, val_to_add)
				seg_min.update(idx_add, val_to_add)
				
				v = nums[j]
				pos = bisect.bisect_left(all_vals, v)
				
				pred = seg_max.query(0, pos)
				succ = seg_min.query(pos, n_val-1)
				
				candidate = INF
				if pred != -INF:
					candidate = min(candidate, abs(v - pred))
				if succ != INF:
					candidate = min(candidate, abs(v - succ))
					
				if candidate < ans:
					ans = candidate
					
			return ans