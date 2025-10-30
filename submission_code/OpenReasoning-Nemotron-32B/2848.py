MOD = 10**9 + 7

class Solution:
	def specialPerm(self, nums: List[int]) -> int:
		n = len(nums)
		graph = [[False] * n for _ in range(n)]
		for i in range(n):
			for j in range(n):
				if i != j:
					if nums[i] % nums[j] == 0 or nums[j] % nums[i] == 0:
						graph[i][j] = True
		
		dp = [[0] * n for _ in range(1 << n)]
		for i in range(n):
			dp[1 << i][i] = 1
		
		for mask in range(1 << n):
			for last in range(n):
				if not (mask & (1 << last)):
					continue
				for nxt in range(n):
					if mask & (1 << nxt):
						continue
					if graph[last][nxt]:
						new_mask = mask | (1 << nxt)
						dp[new_mask][nxt] = (dp[new_mask][nxt] + dp[mask][last]) % MOD
		
		total = 0
		full_mask = (1 << n) - 1
		for i in range(n):
			total = (total + dp[full_mask][i]) % MOD
		return total