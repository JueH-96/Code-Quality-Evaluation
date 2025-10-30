class Solution:
	def differenceOfSums(self, n: int, m: int) -> int:
		total_sum = n * (n + 1) // 2
		k = n // m
		num2 = k * (k + 1) // 2 * m
		return total_sum - 2 * num2