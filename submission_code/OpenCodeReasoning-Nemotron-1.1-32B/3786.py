class Solution:
	def longestPalindromicSubsequence(self, s: str, k: int) -> int:
		n = len(s)
		if n == 0:
			return 0
		dp = [[[0] * n for _ in range(n)] for __ in range(k+1)]
		
		for rem in range(0, k+1):
			for length in range(1, n+1):
				for i in range(0, n - length + 1):
					j = i + length - 1
					if length == 1:
						dp[rem][i][j] = 1
					else:
						skip_i = dp[rem][i+1][j]
						skip_j = dp[rem][i][j-1]
						a = s[i]
						b = s[j]
						diff = abs(ord(a) - ord(b))
						cost_ij = min(diff, 26 - diff)
						if rem >= cost_ij:
							if i+1 <= j-1:
								inner = dp[rem - cost_ij][i+1][j-1]
							else:
								inner = 0
							take_both = 2 + inner
						else:
							take_both = 0
						dp[rem][i][j] = max(skip_i, skip_j, take_both)
		
		return dp[k][0][n-1]