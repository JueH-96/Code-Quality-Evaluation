from typing import List

class Solution:
	def minBitwiseArray(self, nums: List[int]) -> List[int]:
		res = []
		for x in nums:
			if x == 2:
				res.append(-1)
			else:
				v = x + 1
				k_max = 0
				while v % 2 == 0:
					k_max += 1
					v //= 2
				if k_max >= 2:
					k = k_max - 1
					res.append(x - (1 << k))
				else:
					res.append(x - 1)
		return res