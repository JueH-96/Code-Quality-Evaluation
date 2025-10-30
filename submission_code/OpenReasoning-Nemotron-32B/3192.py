class Solution:
	def maximumXorProduct(self, a: int, b: int, n: int) -> int:
		MOD = 10**9 + 7
		x = 0
		for i in range(n-1, -1, -1):
			candidate = x | (1 << i)
			current_prod = (a ^ x) * (b ^ x)
			candidate_prod = (a ^ candidate) * (b ^ candidate)
			if candidate_prod > current_prod:
				x = candidate
		return (a ^ x) * (b ^ x) % MOD