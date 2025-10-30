from typing import List

class Solution:
	def minimumOperations(self, nums: List[int]) -> int:
		total_ops = 0
		for n in nums:
			r = n % 3
			total_ops += min(r, 3 - r)
		return total_ops