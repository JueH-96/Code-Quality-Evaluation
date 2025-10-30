class Solution:
	def stringCount(self, n: int) -> int:
		mod = 10**9 + 7
		a = pow(26, n, mod)
		b = 3 * pow(25, n, mod) % mod
		c = n * pow(25, n-1, mod) % mod
		d = 3 * pow(24, n, mod) % mod
		e = 2 * n * pow(24, n-1, mod) % mod
		f = pow(23, n, mod)
		g = n * pow(23, n-1, mod) % mod
		
		result = (a - b - c + d + e - f - g) % mod
		return result