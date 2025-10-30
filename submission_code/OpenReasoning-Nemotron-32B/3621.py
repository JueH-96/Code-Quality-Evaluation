from typing import List

class Solution:
	def minOperations(self, nums: List[int], k: int) -> int:
		if min(nums) < k:
			return -1
		return len(set(x for x in nums if x > k))