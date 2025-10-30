import math
from functools import reduce

class Solution:
	def maxScore(self, nums: List[int]) -> int:
		n = len(nums)
		if n == 0:
			return 0
		
		g0 = reduce(math.gcd, nums)
		l0 = reduce(lambda a, b: a * b // math.gcd(a, b), nums)
		best = g0 * l0
		
		for i in range(n):
			arr = nums[:i] + nums[i+1:]
			if len(arr) == 0:
				score = 0
			else:
				g = reduce(math.gcd, arr)
				l = reduce(lambda a, b: a * b // math.gcd(a, b), arr)
				score = g * l
			if score > best:
				best = score
				
		return best