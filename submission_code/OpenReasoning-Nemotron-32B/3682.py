class Solution:
	MOD = 10**9 + 7
	max_n = 100000
	fact = None
	inv_fact = None
	
	def precompute(self):
		if Solution.fact is None:
			n = Solution.max_n
			fact_arr = [1] * (n + 1)
			for i in range(1, n + 1):
				fact_arr[i] = fact_arr[i - 1] * i % Solution.MOD
			inv_fact_arr = [1] * (n + 1)
			inv_fact_arr[n] = pow(fact_arr[n], Solution.MOD - 2, Solution.MOD)
			for i in range(n, 0, -1):
				inv_fact_arr[i - 1] = inv_fact_arr[i] * i % Solution.MOD
			Solution.fact = fact_arr
			Solution.inv_fact = inv_fact_arr

	def nCr(self, n, r):
		if r < 0 or r > n:
			return 0
		return Solution.fact[n] * Solution.inv_fact[r] % Solution.MOD * Solution.inv_fact[n - r] % Solution.MOD

	def countGoodArrays(self, n: int, m: int, k: int) -> int:
		self.precompute()
		t = n - k
		binom_val = self.nCr(n - 1, t - 1)
		power_val = pow(m - 1, t - 1, Solution.MOD)
		return m * power_val % Solution.MOD * binom_val % Solution.MOD