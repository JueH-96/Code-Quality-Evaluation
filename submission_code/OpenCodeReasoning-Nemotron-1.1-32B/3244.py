import math
from math import gcd
from functools import reduce

class Solution:
	def minimumArrayLength(self, nums: List[int]) -> int:
		g = reduce(math.gcd, nums)
		a = [x // g for x in nums]
		if 1 in a:
			return 1
		min_val = min(a)
		for x in a:
			if x % min_val != 0:
				return 1
		return 2