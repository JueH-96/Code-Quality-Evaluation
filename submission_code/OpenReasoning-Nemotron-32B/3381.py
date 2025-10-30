from typing import List

class Solution:
	def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
		n = len(nums)
		min_len = n + 1
		for i in range(n):
			cur_or = 0
			for j in range(i, n):
				cur_or |= nums[j]
				if cur_or >= k:
					min_len = min(min_len, j - i + 1)
					break
		return min_len if min_len <= n else -1