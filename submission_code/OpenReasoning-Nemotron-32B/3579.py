from typing import List

class Solution:
	def maxGoodNumber(self, nums: List[int]) -> int:
		a, b, c = nums
		permutations = [
			(a, b, c),
			(a, c, b),
			(b, a, c),
			(b, c, a),
			(c, a, b),
			(c, b, a)
		]
		best = 0
		for p in permutations:
			bin_str = bin(p[0])[2:] + bin(p[1])[2:] + bin(p[2])[2:]
			num_val = int(bin_str, 2)
			if num_val > best:
				best = num_val
		return best