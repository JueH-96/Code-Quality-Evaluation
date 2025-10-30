class Solution:
	def numberOfWays(self, n: int, x: int, y: int) -> int:
		mod = 10**9 + 7
		max_val = 1000
		C_table = [[0] * (max_val + 1) for _ in range(max_val + 1)]
		for i in range(max_val + 1):
			C_table[i][0] = 1
			for j in range(1, i + 1):
				C_table[i][j] = (C_table[i - 1][j] + C_table[i - 1][j - 1]) % mod
		
		total = 0
		k_min = min(n, x)
		for k in range(1, k_min + 1):
			surj = 0
			for i in range(0, k + 1):
				term = C_table[k][i] * pow(k - i, n, mod) % mod
				if i % 2 == 1:
					surj = (surj - term) % mod
				else:
					surj = (surj + term) % mod
			ways_stages = C_table[x][k] * surj % mod
			ways_scores = pow(y, k, mod)
			total = (total + ways_stages * ways_scores) % mod
		
		return total