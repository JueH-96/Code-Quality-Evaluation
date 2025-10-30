MOD = 10**9 + 7

class Solution:
	def specialPerm(self, nums: List[int]) -> int:
		n = len(nums)
		total_mask = (1 << n) - 1
		dp = [[0] * n for _ in range(1 << n)]
		
		for i in range(n):
			dp[1 << i][i] = 1
		
		valid = [[False] * n for _ in range(n)]
		for i in range(n):
			for j in range(n):
				if i != j:
					if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
						valid[i][j] = True
		
		for mask in range(1 << n):
			for last in range(n):
				if (mask >> last) & 1 and dp[mask][last] != 0:
					for j in range(n):
						if not (mask >> j) & 1:
							if valid[last][j]:
								new_mask = mask | (1 << j)
								dp[new_mask][j] = (dp[new_mask][j] + dp[mask][last]) % MOD
		
		return sum(dp[total_mask]) % MOD