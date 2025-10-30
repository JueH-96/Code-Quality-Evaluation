from itertools import permutations

class Solution:
	def maxGoodNumber(self, nums: List[int]) -> int:
		b = [bin(x)[2:] for x in nums]
		return max(int(''.join(p), 2) for p in permutations(b))