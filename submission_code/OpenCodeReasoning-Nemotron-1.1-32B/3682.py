mod = 10**9 + 7

class Solution:
	def countGoodArrays(self, n: int, m: int, k: int) -> int:
		mod = 10**9 + 7
		r = n - k
		if r < 1:
			return 0
		N = n - 1
		if N < 0:
			return 0
		fact = [1] * (N + 1)
		for i in range(1, N + 1):
			fact[i] = fact[i - 1] * i % mod
		
		inv_fact = [1] * (N + 1)
		if N > 0:
			inv_fact[N] = pow(fact[N], mod - 2, mod)
			for i in range(N, 0, -1):
				inv_fact[i - 1] = inv_fact[i] * i % mod
		
		if r - 1 < 0 or r - 1 > N:
			comb_val = 0
		else:
			comb_val = fact[N] * inv_fact[r - 1] % mod * inv_fact[N - (r - 1)] % mod
		
		color_part = m * pow(m - 1, r - 1, mod) % mod
		return comb_val * color_part % mod