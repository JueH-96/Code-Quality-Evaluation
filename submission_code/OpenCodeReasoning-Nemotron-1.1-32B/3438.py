class Solution:
	class FenwickTree:
		def __init__(self, n):
			self.n = n
			self.tree = [0] * (n + 1)
		
		def update(self, index, delta):
			i = index + 1
			while i <= self.n:
				self.tree[i] += delta
				i += i & -i
		
		def query(self, index):
			if index < 0:
				return 0
			res = 0
			i = index + 1
			while i:
				res += self.tree[i]
				i -= i & -i
			return res
		
		def range_query(self, l, r):
			if l > r:
				return 0
			return self.query(r) - self.query(l - 1)
	
	def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
		n = len(nums)
		fenw = self.FenwickTree(n)
		A = [0] * n
		
		for i in range(1, n - 1):
			if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
				A[i] = 1
				fenw.update(i, 1)
		
		ans = []
		for q in queries:
			if q[0] == 1:
				l, r = q[1], q[2]
				if l + 1 <= r - 1:
					res = fenw.range_query(l + 1, r - 1)
				else:
					res = 0
				ans.append(res)
			else:
				idx = q[1]
				val = q[2]
				if nums[idx] == val:
					continue
				old_val = nums[idx]
				nums[idx] = val
				for j in [idx - 1, idx, idx + 1]:
					if j < 1 or j > n - 2:
						continue
					new_peak = 1 if (nums[j] > nums[j - 1] and nums[j] > nums[j + 1]) else 0
					if A[j] != new_peak:
						fenw.update(j, new_peak - A[j])
						A[j] = new_peak
		return ans