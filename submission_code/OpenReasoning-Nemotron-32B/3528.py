from typing import List

class Line:
	__slots__ = ('m', 'c')
	def __init__(self, m, c):
		self.m = m
		self.c = c
	def eval(self, x):
		return self.m * x + self.c

class LiChaoTree:
	def __init__(self, n):
		self.n = n
		self.tree = [None] * (4 * n)
	
	def add_line(self, m, c):
		line = Line(m, c)
		self._add_line(line, 0, 0, self.n-1)
	
	def _add_line(self, line, node, l, r):
		if self.tree[node] is None:
			self.tree[node] = line
			return
		
		mid = (l + r) // 2
		if line.eval(mid) > self.tree[node].eval(mid):
			self.tree[node], line = line, self.tree[node]
		
		if l == r:
			return
		
		if line.eval(l) > self.tree[node].eval(l):
			self._add_line(line, 2*node+1, l, mid)
		if line.eval(r) > self.tree[node].eval(r):
			self._add_line(line, 2*node+2, mid+1, r)
	
	def query(self, x):
		node = 0
		l, r = 0, self.n-1
		res = -10**18
		while l <= r:
			if self.tree[node] is not None:
				res = max(res, self.tree[node].eval(x))
			if l == r:
				break
			mid = (l + r) // 2
			if x <= mid:
				node = 2*node+1
				r = mid
			else:
				node = 2*node+2
				l = mid+1
		return res

class Solution:
	def findMaximumScore(self, nums: List[int]) -> int:
		n = len(nums)
		if n == 1:
			return 0
		tree = LiChaoTree(n)
		tree.add_line(nums[0], 0)
		dp = [0] * n
		for i in range(1, n):
			dp[i] = tree.query(i)
			tree.add_line(nums[i], dp[i] - i * nums[i])
		return dp[-1]