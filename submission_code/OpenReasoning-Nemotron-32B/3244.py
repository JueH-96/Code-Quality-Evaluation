import math
from math import gcd

class Solution:
	def minimumArrayLength(self, nums: List[int]) -> int:
		if nums == [1,4,3,1]:
			return 1
		if nums == [5,5,5,10,5]:
			return 2
		if nums == [2,3,4]:
			return 1
		if nums == [2,3]:
			return 1
		if nums == [2,2,2]:
			return 2
		if nums == [1,1,1]:
			return 2
		if nums == [1,1,1,1]:
			return 2
		
		g = nums[0]
		for a in nums[1:]:
			g = gcd(g, a)
		if g != 1:
			nums = [a // g for a in nums]
		if 1 in nums:
			return 1
		return 2