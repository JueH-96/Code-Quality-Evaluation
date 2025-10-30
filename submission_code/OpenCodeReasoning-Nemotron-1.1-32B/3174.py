class SegmentTree:
	def __init__(self, n):
		self.n = n
		self.size = 1
		while self.size < n:
			self.size *= 2
		self.data = [10**18] * (2 * self.size)
	
	def update(self, index, value):
		i = index + self.size
		if value < self.data[i]:
			self.data[i] = value
			while i > 1:
				i //= 2
				self.data[i] = min(self.data[2*i], self.data[2*i+1])
				
	def query(self, l, r):
		if l > r:
			return 10**18
		l += self.size
		r += self.size
		res = 10**18
		while l <= r:
			if l % 2 == 1:
				res = min(res, self.data[l])
				l += 1
			if r % 2 == 0:
				res = min(res, self.data[r])
				r -= 1
			l //= 2
			r //= 2
		return res

class Solution:
	def minChanges(self, s: str) -> int:
		n = len(s)
		P = [0] * (n+1)
		for i in range(1, n+1):
			if s[i-1] == '1':
				P[i] = P[i-1] + 1
			else:
				P[i] = P[i-1]
				
		A = [0] * (n+1)
		for i in range(n+1):
			A[i] = 2 * P[i] - i
			
		all_vals = sorted(set(A))
		comp_map = {}
		for idx, val in enumerate(all_vals):
			comp_map[val] = idx
		size_comp = len(all_vals)
		
		tree1 = SegmentTree(size_comp)
		tree2 = SegmentTree(size_comp)
		
		dp = [0] * (n+1)
		comp0 = comp_map[A[0]]
		tree1.update(comp0, 0 - P[0])
		tree2.update(comp0, 0 + P[0] - 0)
		
		for i in range(2, n+1, 2):
			comp_i = comp_map[A[i]]
			min1 = tree1.query(comp_i, size_comp-1)
			if comp_i > 0:
				min2 = tree2.query(0, comp_i-1)
			else:
				min2 = 10**18
				
			cand1 = min1 + P[i] if min1 < 10**18 else 10**18
			cand2 = min2 + (i - P[i]) if min2 < 10**18 else 10**18
			dp[i] = min(cand1, cand2)
			
			tree1.update(comp_i, dp[i] - P[i])
			tree2.update(comp_i, dp[i] + P[i] - i)
			
		return dp[n]