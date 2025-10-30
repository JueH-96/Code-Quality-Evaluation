class Solution:
	def numberOfChild(self, n: int, k: int) -> int:
		cycle = 2 * (n - 1)
		k_mod = k % cycle
		if k_mod < n:
			return k_mod
		else:
			return 2 * (n - 1) - k_mod