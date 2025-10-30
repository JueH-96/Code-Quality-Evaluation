MOD = 10**9 + 7

class Solution:
	def maxSum(self, nums: List[int], k: int) -> int:
		max_bit = 31
		max_exp = 60
		pow2 = [1] * (max_exp + 1)
		for i in range(1, max_exp + 1):
			pow2[i] = (pow2[i-1] * 2) % MOD
		
		c = [0] * max_bit
		for num in nums:
			for j in range(max_bit):
				if num & (1 << j):
					c[j] += 1
		
		total = 0
		for j in range(max_bit):
			count_j = min(c[j], k)
			exp = 2 * j
			total = (total + pow2[exp] * count_j) % MOD
		
		for j in range(max_bit):
			for j2 in range(j + 1, max_bit):
				count_jj2 = min(c[j], c[j2], k)
				exp = j + j2
				term = pow2[exp] * count_jj2 % MOD
				term = (2 * term) % MOD
				total = (total + term) % MOD
		
		return total