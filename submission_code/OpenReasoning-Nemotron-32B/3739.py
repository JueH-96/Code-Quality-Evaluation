class Solution:
	fact = None
	inv_fact = None

	def distanceSum(self, m: int, n: int, k: int) -> int:
		MOD = 10**9 + 7
		total_cells = m * n
		
		if Solution.fact is None:
			max_n = 100000
			fact_arr = [1] * (max_n + 1)
			inv_fact_arr = [1] * (max_n + 1)
			for i in range(1, max_n + 1):
				fact_arr[i] = fact_arr[i-1] * i % MOD
			inv_fact_arr[max_n] = pow(fact_arr[max_n], MOD-2, MOD)
			for i in range(max_n, 0, -1):
				inv_fact_arr[i-1] = inv_fact_arr[i] * i % MOD
			Solution.fact = fact_arr
			Solution.inv_fact = inv_fact_arr
		
		if k < 2 or k > total_cells:
			return 0
		
		N = total_cells - 2
		r = k - 2
		if r < 0 or r > N:
			binom_val = 0
		else:
			binom_val = Solution.fact[N] * Solution.inv_fact[r] % MOD * Solution.inv_fact[N - r] % MOD
		
		inv6 = pow(6, MOD-2, MOD)
		
		S_row = (m - 1) % MOD
		S_row = S_row * m % MOD
		S_row = S_row * (m + 1) % MOD
		S_row = S_row * inv6 % MOD
		
		S_col = (n - 1) % MOD
		S_col = S_col * n % MOD
		S_col = S_col * (n + 1) % MOD
		S_col = S_col * inv6 % MOD
		
		total_x = (n * n) % MOD * S_row % MOD
		total_y = (m * m) % MOD * S_col % MOD
		
		ans = (total_x + total_y) % MOD * binom_val % MOD
		return ans