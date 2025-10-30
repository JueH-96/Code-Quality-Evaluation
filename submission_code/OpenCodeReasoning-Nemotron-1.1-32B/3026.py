class Solution:
	def minimumPossibleSum(self, n: int, target: int) -> int:
		MOD = 10**9 + 7
		A = min(n, target // 2)
		B = n - A
		total = A * (A + 1) // 2 + B * target + B * (B - 1) // 2
		return total % MOD