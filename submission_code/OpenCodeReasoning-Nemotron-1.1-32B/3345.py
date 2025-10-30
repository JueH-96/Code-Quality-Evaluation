mod = 10**9 + 7

class Solution:
	def sumOfPower(self, nums: List[int], k: int) -> int:
		n = len(nums)
		dp = [[0] * (n + 1) for _ in range(k + 1)]
		dp[0][0] = 1
		
		for num in nums:
			for s in range(k, -1, -1):
				for c in range(n, 0, -1):
					if s >= num:
						dp[s][c] = (dp[s][c] + dp[s - num][c - 1]) % mod
		
		ans = 0
		for c in range(0, n + 1):
			ans = (ans + dp[k][c] * pow(2, n - c, mod)) % mod
		
		return ans