from typing import List

class Solution:
	def countWays(self, nums: List[int]) -> int:
		n = len(nums)
		freq = [0] * (n + 1)
		for num in nums:
			if num < n:
				freq[num] += 1
		
		F = [0] * (n + 1)
		for k in range(1, n + 1):
			F[k] = F[k - 1] + freq[k - 1]
		
		ans = 0
		for k in range(0, n + 1):
			if k < n and freq[k] > 0:
				continue
			if F[k] == k:
				ans += 1
				
		return ans