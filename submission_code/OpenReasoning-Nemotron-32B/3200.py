MOD = 10**9 + 7

class Solution:
	def stringCount(self, n: int) -> int:
		if n < 4:
			return 0
		total = pow(26, n, MOD)
		t1 = 3 * pow(25, n, MOD) % MOD
		t2 = n * pow(25, n-1, MOD) % MOD
		t3 = 3 * pow(24, n, MOD) % MOD
		t4 = (2 * n) * pow(24, n-1, MOD) % MOD
		t5 = pow(23, n, MOD)
		t6 = n * pow(23, n-1, MOD) % MOD
		
		bad = (t1 + t2 - t3 - t4 + t5 + t6) % MOD
		good = (total - bad) % MOD
		return good