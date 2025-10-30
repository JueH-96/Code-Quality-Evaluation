class Solution:
	def numberOfGoodPartitions(self, nums: List[int]) -> int:
		mod = 10**9 + 7
		n = len(nums)
		last_occurrence = {}
		for i in range(n-1, -1, -1):
			if nums[i] not in last_occurrence:
				last_occurrence[nums[i]] = i
		
		dp = [0] * (n + 1)
		dp[0] = 1
		total = 1
		max_last = 0
		
		for i in range(n):
			num = nums[i]
			max_last = max(max_last, last_occurrence[num])
			
			if i < max_last:
				dp[i+1] = 0
			else:
				dp[i+1] = total
				total = (total + dp[i+1]) % mod
				max_last = 0
				
		return dp[n] % mod