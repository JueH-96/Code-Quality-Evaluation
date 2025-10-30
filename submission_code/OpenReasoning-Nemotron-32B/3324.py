from collections import Counter

class Solution:
	def isPossibleToSplit(self, nums: List[int]) -> bool:
		freq = Counter(nums)
		for count in freq.values():
			if count > 2:
				return False
		return True