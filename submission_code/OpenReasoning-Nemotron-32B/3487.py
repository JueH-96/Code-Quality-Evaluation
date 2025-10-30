from typing import List

class Solution:
	def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
		n = len(source)
		m = len(pattern)
		target_set = set(targetIndices)
		dp = [[-10**9] * (m + 1) for _ in range(n + 1)]
		dp[0][0] = 0
		
		for i in range(n):
			for j in range(m + 1):
				if dp[i][j] < 0:
					continue
				if i in target_set:
					if dp[i + 1][j] < dp[i][j] + 1:
						dp[i + 1][j] = dp[i][j] + 1
					if dp[i + 1][j] < dp[i][j]:
						dp[i + 1][j] = dp[i][j]
					if j < m and source[i] == pattern[j]:
						if dp[i + 1][j + 1] < dp[i][j]:
							dp[i + 1][j + 1] = dp[i][j]
				else:
					if dp[i + 1][j] < dp[i][j]:
						dp[i + 1][j] = dp[i][j]
					if j < m and source[i] == pattern[j]:
						if dp[i + 1][j + 1] < dp[i][j]:
							dp[i + 1][j + 1] = dp[i][j]
		
		return dp[n][m]