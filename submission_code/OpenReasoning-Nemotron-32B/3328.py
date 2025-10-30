import math

class Solution:
	def minOperations(self, k: int) -> int:
		if k == 1:
			return 0
		sqrt_k = math.isqrt(k)
		n_max = min(k, 2 * sqrt_k + 100)
		ans = float('inf')
		for x in range(1, n_max + 1):
			d = (k + x - 1) // x
			candidate = x + d - 2
			if candidate < ans:
				ans = candidate
		return ans