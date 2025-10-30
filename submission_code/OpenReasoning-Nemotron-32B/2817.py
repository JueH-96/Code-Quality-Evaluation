class Solution:
	def minimumCost(self, s: str) -> int:
		n = len(s)
		return min(self.solve_for_target(s, n, '0'), self.solve_for_target(s, n, '1'))
	
	def solve_for_target(self, s, n, target):
		dp = [[0] * 2 for _ in range(2)]
		for i in range(n-1, -1, -1):
			new_dp = [[10**18] * 2 for _ in range(2)]
			for p in range(2):
				for s_val in range(2):
					total_flip = p + s_val
					if total_flip % 2 == 0:
						effective_char = s[i]
					else:
						effective_char = '1' if s[i] == '0' else '0'
					
					if effective_char == target:
						cost = dp[p][s_val]
					else:
						cost = min(
							(i+1) + dp[p][s_val],
							(n - i) + dp[p][s_val ^ 1]
						)
					new_dp[p][s_val] = cost
			dp = new_dp
		return dp[0][0]