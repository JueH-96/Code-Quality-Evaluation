class Solution:
	def minimumPossibleSum(self, n: int, target: int) -> int:
		mod = 10**9 + 7
		k = target // 2
		if n <= k:
			total = n * (n + 1) // 2
		else:
			total = k * (k + 1) // 2
			total += (n - k) * target
			total += (n - k) * (n - k - 1) // 2
		return total % mod