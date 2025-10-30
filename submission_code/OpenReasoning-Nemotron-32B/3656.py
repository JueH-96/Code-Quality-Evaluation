from typing import List

class Solution:
	def minimumOperations(self, nums: List[int]) -> int:
		n = len(nums)
		max_ops = (n + 2) // 3
		for k in range(max_ops + 1):
			start_index = 3 * k
			if start_index >= n:
				return k
			seen = set()
			has_duplicate = False
			for i in range(start_index, n):
				if nums[i] in seen:
					has_duplicate = True
					break
				seen.add(nums[i])
			if not has_duplicate:
				return k