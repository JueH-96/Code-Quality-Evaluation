class Solution:
	def maxScore(self, a: List[int], b: List[int]) -> int:
		n = len(b)
		NEG_INF = -10**18
		
		dp = [a[0] * x for x in b]
		
		M_prev = [0] * n
		if n > 0:
			M_prev[0] = dp[0]
			for i in range(1, n):
				M_prev[i] = max(M_prev[i-1], dp[i])
		
		dp = [NEG_INF] * n
		for i in range(1, n):
			dp[i] = M_prev[i-1] + a[1] * b[i]
		
		M_prev = [NEG_INF] * n
		if n > 0:
			M_prev[0] = dp[0]
			for i in range(1, n):
				M_prev[i] = max(M_prev[i-1], dp[i])
		
		dp = [NEG_INF] * n
		for i in range(2, n):
			dp[i] = M_prev[i-1] + a[2] * b[i]
		
		M_prev = [NEG_INF] * n
		if n > 0:
			M_prev[0] = dp[0]
			for i in range(1, n):
				M_prev[i] = max(M_prev[i-1], dp[i])
		
		dp = [NEG_INF] * n
		for i in range(3, n):
			dp[i] = M_prev[i-1] + a[3] * b[i]
		
		return max(dp)