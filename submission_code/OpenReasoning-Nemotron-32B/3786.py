class Solution:
	def longestPalindromicSubsequence(self, s: str, k: int) -> int:
		n = len(s)
		dp = [[[0] * (k+1) for _ in range(n)] for __ in range(n)]
		
		for i in range(n):
			for r in range(k+1):
				dp[i][i][r] = 1
		
		for length in range(2, n+1):
			for i in range(0, n - length + 1):
				j = i + length - 1
				for r in range(0, k+1):
					candidate1 = dp[i+1][j][r]
					candidate2 = dp[i][j-1][r]
					best = max(candidate1, candidate2)
					
					diff = abs(ord(s[i]) - ord(s[j]))
					cost = min(diff, 26 - diff)
					if r >= cost:
						if i+1 <= j-1:
							inner_val = dp[i+1][j-1][r - cost]
						else:
							inner_val = 0
						candidate3 = 2 + inner_val
						if candidate3 > best:
							best = candidate3
					dp[i][j][r] = best
		
		return dp[0][n-1][k]