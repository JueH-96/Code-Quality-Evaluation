from typing import List

class Solution:
	def minimumAverage(self, nums: List[int]) -> float:
		min_avg = float('inf')
		n = len(nums)
		for _ in range(n // 2):
			min_val = min(nums)
			max_val = max(nums)
			nums.remove(min_val)
			nums.remove(max_val)
			avg = (min_val + max_val) / 2.0
			if avg < min_avg:
				min_avg = avg
		return min_avg