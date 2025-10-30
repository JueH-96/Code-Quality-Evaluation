from typing import List

class Solution:
	def maxSubarrays(self, nums: List[int]) -> int:
		n = len(nums)
		total_and = nums[0]
		for i in range(1, n):
			total_and &= nums[i]
		if total_and != 0:
			return 1
		
		dp = [-10**9] * n
		and_map = {}
		
		and_map = { nums[0]: 0 }
		if nums[0] == 0:
			dp[0] = 1
		else:
			dp[0] = -10**9
		
		for i in range(1, n):
			new_and_map = {}
			
			if dp[i-1] != -10**9:
				x = nums[i]
				if x in new_and_map:
					if dp[i-1] > new_and_map[x]:
						new_and_map[x] = dp[i-1]
				else:
					new_and_map[x] = dp[i-1]
			
			for key, base_val in and_map.items():
				x = key & nums[i]
				if x in new_and_map:
					if base_val > new_and_map[x]:
						new_and_map[x] = base_val
				else:
					new_and_map[x] = base_val
			
			if 0 in new_and_map:
				dp[i] = new_and_map[0] + 1
			else:
				dp[i] = -10**9
			
			and_map = new_and_map
		
		return dp[n-1]