class Solution:
	mod = 10**9 + 7
	max_val = 1000
	C_table = None
	
	def __init__(self):
		if Solution.C_table is None:
			C_table = [[0] * (Solution.max_val + 1) for _ in range(Solution.max_val + 1)]
			for i in range(Solution.max_val + 1):
				C_table[i][0] = 1
				for j in range(1, i + 1):
					C_table[i][j] = (C_table[i-1][j-1] + C_table[i-1][j]) % Solution.mod
			Solution.C_table = C_table

	def numberOfWays(self, n: int, x: int, y: int) -> int:
		total = 0
		k_max = min(n, x)
		for k in range(1, k_max + 1):
			inner = 0
			for i in range(0, k + 1):
				base = k - i
				term = Solution.C_table[k][i] * pow(base, n, Solution.mod) % Solution.mod
				if i % 2 == 1:
					inner = (inner - term) % Solution.mod
				else:
					inner = (inner + term) % Solution.mod
			inner %= Solution.mod
			if inner < 0:
				inner += Solution.mod
			ways = Solution.C_table[x][k] * inner % Solution.mod
			ways = ways * pow(y, k, Solution.mod) % Solution.mod
			total = (total + ways) % Solution.mod
		return total