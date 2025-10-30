class Solution:
	def minOperations(self, s1: str, s2: str, x: int) -> int:
		n = len(s1)
		d = [i for i in range(n) if s1[i] != s2[i]]
		m = len(d)
		if m % 2 != 0:
			return -1
		if m == 0:
			return 0
		
		dp = [[0] * m for _ in range(m)]
		
		for i in range(m-1):
			dp[i][i+1] = min(x, d[i+1] - d[i])
		
		for L in range(4, m+1, 2):
			for i in range(0, m - L + 1):
				j = i + L - 1
				dp[i][j] = float('inf')
				for k in range(i, j, 2):
					cost_pair = min(x, d[j] - d[k])
					left_val = dp[i][k-1] if k-1 >= i else 0
					right_val = dp[k+1][j-1] if k+1 <= j-1 else 0
					total = cost_pair + left_val + right_val
					if total < dp[i][j]:
						dp[i][j] = total
		return dp[0][m-1]