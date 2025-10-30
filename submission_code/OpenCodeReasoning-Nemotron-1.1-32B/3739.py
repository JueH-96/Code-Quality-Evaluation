class Solution:
	fact = None
	inv_fact = None
	maxN = 100000

	@classmethod
	def precompute(cls):
		mod = 10**9 + 7
		n = cls.maxN
		cls.fact = [1] * (n + 1)
		cls.inv_fact = [1] * (n + 1)
		for i in range(1, n + 1):
			cls.fact[i] = cls.fact[i - 1] * i % mod
		cls.inv_fact[n] = pow(cls.fact[n], mod - 2, mod)
		for i in range(n, 0, -1):
			cls.inv_fact[i - 1] = cls.inv_fact[i] * i % mod

	def distanceSum(self, m: int, n: int, k: int) -> int:
		mod = 10**9 + 7
		if Solution.fact is None:
			Solution.precompute()
		
		total_cells = m * n
		N = total_cells - 2
		r = k - 2
		
		if r < 0 or r > N:
			binom = 0
		else:
			binom = Solution.fact[N] * Solution.inv_fact[r] % mod * Solution.inv_fact[N - r] % mod
		
		inner_x = 0
		if m > 1:
			inner_x = m * (m - 1) * (m + 1) // 6
		
		inner_y = 0
		if n > 1:
			inner_y = n * (n - 1) * (n + 1) // 6
		
		Sx = (n * n) % mod
		Sx = Sx * (inner_x % mod) % mod
		
		Sy = (m * m) % mod
		Sy = Sy * (inner_y % mod) % mod
		
		total = binom * ((Sx + Sy) % mod) % mod
		return total