MOD = 10**9 + 7

def precompute_combinations(n_max):
	C = [[0] * (n_max + 1) for _ in range(n_max + 1)]
	for i in range(n_max + 1):
		C[i][0] = 1
		for j in range(1, i + 1):
			C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % MOD
	return C

class Solution:
	def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
		req_arr = [-1] * n
		for end, cnt in requirements:
			if end < n:
				req_arr[end] = cnt
		
		n_max = 300
		C_table = precompute_combinations(n_max)
		
		F = [[0] * 401 for _ in range(n + 1)]
		for i in range(0, n + 1):
			max_t = min(i, 400)
			for t in range(0, max_t + 1):
				x_low = max(0, i - t)
				x_high = n - 1 - t
				if x_low > x_high:
					continue
				for x in range(x_low, x_high + 1):
					term1 = C_table[x][i - t]
					term2 = C_table[n - 1 - x][t]
					F[i][t] = (F[i][t] + term1 * term2) % MOD
		
		dp = [[0] * 401 for _ in range(n + 1)]
		dp[0][0] = 1
		
		for i in range(0, n):
			for j in range(0, 401):
				if dp[i][j] == 0:
					continue
				max_t = min(i, 400)
				for t in range(0, max_t + 1):
					new_j = j + t
					if new_j > 400:
						break
					comb = C_table[n][i]
					inv_comb = pow(comb, MOD - 2, MOD)
					add_val = dp[i][j] * F[i][t] % MOD
					add_val = add_val * inv_comb % MOD
					dp[i + 1][new_j] = (dp[i + 1][new_j] + add_val) % MOD
			if i < n:
				if req_arr[i] != -1:
					for j in range(0, 401):
						if j != req_arr[i]:
							dp[i + 1][j] = 0
		
		ans = 0
		for j in range(0, 401):
			ans = (ans + dp[n][j]) % MOD
		return ans