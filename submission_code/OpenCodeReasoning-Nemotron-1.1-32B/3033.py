class Solution:
	def minOperations(self, s1: str, s2: str, x: int) -> int:
		n_str = len(s1)
		A = []
		for i in range(n_str):
			if s1[i] != s2[i]:
				A.append(i)
		n = len(A)
		if n % 2 != 0:
			return -1
		if n == 0:
			return 0
		
		dp = [[10**9] * n for _ in range(n)]
		
		for length in range(2, n+1, 2):
			for i in range(0, n - length + 1):
				j = i + length - 1
				for k in range(i+1, j+1, 2):
					cost_pair = min(x, A[k] - A[i])
					cost_mid = 0
					if k - 1 >= i + 1:
						cost_mid += dp[i+1][k-1]
					if j >= k + 1:
						cost_mid += dp[k+1][j]
					total = cost_pair + cost_mid
					if total < dp[i][j]:
						dp[i][j] = total
		return dp[0][n-1]