MOD = 10**9 + 7

class Solution:
	def countOfPairs(self, nums: List[int]) -> int:
		n = len(nums)
		dp = [0] * 51
		for x in range(nums[0] + 1):
			dp[x] = 1
		
		for i in range(1, n):
			prefix = [0] * 51
			prefix[0] = dp[0]
			for j in range(1, 51):
				prefix[j] = (prefix[j-1] + dp[j]) % MOD
			
			new_dp = [0] * 51
			d = max(0, nums[i] - nums[i-1])
			for y in range(nums[i] + 1):
				if y < d:
					new_dp[y] = 0
				else:
					x_max = min(nums[i-1], y - d)
					new_dp[y] = prefix[x_max]
			dp = new_dp
		
		total = 0
		for x in range(nums[-1] + 1):
			total = (total + dp[x]) % MOD
		return total