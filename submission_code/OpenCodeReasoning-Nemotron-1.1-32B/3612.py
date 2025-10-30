from typing import List

class Solution:
	def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
		n = len(nums)
		valid = [True] * (n - k + 1)
		for i in range(n - k + 1):
			for j in range(i, i + k - 1):
				if nums[j] >= nums[j + 1]:
					valid[i] = False
					break
		for i in range(n - 2 * k + 1):
			if valid[i] and valid[i + k]:
				return True
		return False