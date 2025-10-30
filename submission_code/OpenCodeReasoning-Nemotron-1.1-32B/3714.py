MOD = 10**9 + 7

class Solution:
	def minMaxSums(self, nums: List[int], k: int) -> int:
		n = len(nums)
		k0 = k - 1
		
		if k0 < 0:
			S_arr = [0] * (n + 1)
		else:
			if k0 == 0:
				S_arr = [1] * (n + 1)
			else:
				dp = [0] * (k0 + 1)
				dp[0] = 1
				S_arr = [0] * (n + 1)
				S_arr[0] = 1
				for i in range(1, n + 1):
					j_end = min(i, k0)
					for j in range(j_end, 0, -1):
						dp[j] = (dp[j] + dp[j - 1]) % MOD
					total_sum = 0
					for j in range(0, j_end + 1):
						total_sum = (total_sum + dp[j]) % MOD
					S_arr[i] = total_sum
		
		nums.sort()
		total_max = 0
		total_min = 0
		for i in range(n):
			total_max = (total_max + nums[i] * S_arr[i]) % MOD
			total_min = (total_min + nums[i] * S_arr[n - 1 - i]) % MOD
		
		return (total_max + total_min) % MOD