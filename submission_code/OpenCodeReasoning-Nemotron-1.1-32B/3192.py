MOD = 10**9 + 7

class Solution:
	def maximumXorProduct(self, a: int, b: int, n: int) -> int:
		if n == 0:
			return (a * b) % MOD
		max_val = 0
		for x in range(0, 1 << n):
			product_val = (a ^ x) * (b ^ x)
			if product_val > max_val:
				max_val = product_val
		return max_val % MOD