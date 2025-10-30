from collections import defaultdict
from typing import List

class Solution:
	def minimumSeconds(self, nums: List[int]) -> int:
		n = len(nums)
		groups = defaultdict(list)
		for i, num in enumerate(nums):
			groups[num].append(i)
		
		ans = float('inf')
		for key, positions in groups.items():
			m = len(positions)
			gaps = []
			for i in range(m):
				if i == m - 1:
					gap = positions[0] + n - positions[-1] - 1
				else:
					gap = positions[i+1] - positions[i] - 1
				gaps.append(gap)
			max_gap = max(gaps)
			if max_gap == 0:
				return 0
			candidate = (max_gap + 1) // 2
			if candidate < ans:
				ans = candidate
		return ans