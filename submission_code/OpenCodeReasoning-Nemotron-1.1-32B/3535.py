mod = 10**9 + 7

class Solution:
	def countOfPairs(self, nums: List[int]) -> int:
		n = len(nums)
		dp = [1] * (nums[0] + 1)
		
		for i in range(1, n):
			prev_len = len(dp)
			prefix = [0] * prev_len
			prefix[0] = dp[0]
			for j in range(1, prev_len):
				prefix[j] = (prefix[j-1] + dp[j]) % mod
			
			d = nums[i] - nums[i-1]
			if d < 0:
				d = 0
			
			new_dp = [0] * (nums[i] + 1)
			
			for x in range(0, nums[i] + 1):
				high_prev = x - d
				if high_prev < 0:
					new_dp[x] = 0
				else:
					high_prev = min(high_prev, prev_len - 1)
					new_dp[x] = prefix[high_prev]
			
			dp = new_dp
		
		ans = sum(dp) % mod
		return ans