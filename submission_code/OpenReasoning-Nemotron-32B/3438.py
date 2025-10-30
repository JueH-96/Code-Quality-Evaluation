from typing import List

class Solution:
	class Fenw:
		def __init__(self, size):
			self.n = size
			self.tree = [0] * (self.n + 1)
		
		def update(self, index, delta):
			i = index + 1
			while i <= self.n:
				self.tree[i] += delta
				i += i & -i
		
		def query_range(self, l, r):
			if l > r:
				return 0
			return self._prefix(r) - self._prefix(l - 1)
		
		def _prefix(self, index):
			if index < 0:
				return 0
			i = index + 1
			s = 0
			while i:
				s += self.tree[i]
				i -= i & -i
			return s

	def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
		n = len(nums)
		fenw = self.Fenw(n)
		peaks = [0] * n
		
		for i in range(1, n - 1):
			if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
				peaks[i] = 1
				fenw.update(i, 1)
		
		ans = []
		for q in queries:
			if q[0] == 1:
				l, r = q[1], q[2]
				start = l + 1
				end = r - 1
				if start > end:
					ans.append(0)
				else:
					total = fenw.query_range(start, end)
					ans.append(total)
			else:
				idx = q[1]
				val = q[2]
				nums[idx] = val
				indices_to_update = set()
				for pos in [idx - 1, idx, idx + 1]:
					if 1 <= pos <= n - 2:
						indices_to_update.add(pos)
				
				for i in indices_to_update:
					if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
						new_peak = 1
					else:
						new_peak = 0
					delta = new_peak - peaks[i]
					if delta != 0:
						fenw.update(i, delta)
						peaks[i] = new_peak
		
		return ans