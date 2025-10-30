from typing import List

class Solution:
	def checkArray(self, nums: List[int], k: int) -> bool:
		n = len(nums)
		diff = [0] * (n + 1)
		current = 0
		for i in range(n):
			current += diff[i]
			if nums[i] < current:
				return False
			if nums[i] > current:
				if i + k > n:
					return False
				x = nums[i] - current
				current += x
				diff[i + k] -= x
		return True