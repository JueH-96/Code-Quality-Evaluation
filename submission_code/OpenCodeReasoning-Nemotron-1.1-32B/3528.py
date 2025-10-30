class LiChaoTree:
	def __init__(self, size):
		self.n = size
		self.tree = [None] * (4 * self.n)
	
	def add_line(self, m, b):
		self._add_line(0, 0, self.n-1, m, b)
		
	def _add_line(self, node, l, r, m, b):
		if l > r:
			return
		current = self.tree[node]
		if current is None:
			self.tree[node] = (m, b)
			return
			
		m0, b0 = current
		mid = (l + r) // 2
		f_mid = m * mid + b
		g_mid = m0 * mid + b0
		
		if f_mid > g_mid:
			self.tree[node] = (m, b)
			m, b = m0, b0
			m0, b0 = self.tree[node]
		
		f_l = m * l + b
		g_l = m0 * l + b0
		if f_l > g_l:
			self._add_line(2*node+1, l, mid, m, b)
		else:
			f_r = m * r + b
			g_r = m0 * r + b0
			if f_r > g_r:
				self._add_line(2*node+2, mid+1, r, m, b)
				
	def query(self, x):
		return self._query(0, 0, self.n-1, x)
		
	def _query(self, node, l, r, x):
		if l > r:
			return -10**18
		res = -10**18
		if self.tree[node] is not None:
			m0, b0 = self.tree[node]
			res = m0 * x + b0
			
		if l == r:
			return res
			
		mid = (l + r) // 2
		if x <= mid:
			res2 = self._query(2*node+1, l, mid, x)
		else:
			res2 = self._query(2*node+2, mid+1, r, x)
		return max(res, res2)

class Solution:
	def findMaximumScore(self, nums: List[int]) -> int:
		n = len(nums)
		if n == 1:
			return 0
			
		size = 1
		while size < n:
			size *= 2
			
		tree = LiChaoTree(size)
		tree.add_line(nums[0], 0)
		
		dp = [0] * n
		dp[0] = 0
		
		for j in range(1, n):
			dp[j] = tree.query(j)
			A_j = dp[j] - j * nums[j]
			tree.add_line(nums[j], A_j)
			
		return dp[n-1]