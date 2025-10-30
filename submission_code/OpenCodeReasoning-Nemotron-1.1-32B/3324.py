from collections import Counter

class Solution:
	def isPossibleToSplit(self, nums: List[int]) -> bool:
		cnt = Counter(nums)
		for count in cnt.values():
			if count > 2:
				return False
		return True