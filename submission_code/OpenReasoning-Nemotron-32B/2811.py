class Solution:
	def minimumSum(self, n: int, k: int) -> int:
		low_set_size = (k - 1) // 2
		if n <= low_set_size:
			return n * (n + 1) // 2
		else:
			total = low_set_size * (low_set_size + 1) // 2
			remaining = n - low_set_size
			if k % 2 == 0:
				if remaining > 0:
					total += k // 2
					remaining -= 1
			if remaining > 0:
				total += k * remaining + (remaining - 1) * remaining // 2
			return total