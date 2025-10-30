from typing import List

class Solution:
	def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
		n = len(nums)
		prefix = [0] * (n + 1)
		for i in range(1, n + 1):
			prefix[i] = prefix[i - 1] + nums[i - 1]
		
		min_sum = float('inf')
		
		for i in range(n):
			start_j = i + l
			end_j = min(i + r, n)
			for j in range(start_j, end_j + 1):
				s = prefix[j] - prefix[i]
				if s > 0:
					if s < min_sum:
						min_sum = s
		
		return min_sum if min_sum != float('inf') else -1