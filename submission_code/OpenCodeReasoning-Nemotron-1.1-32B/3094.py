from collections import Counter

class Solution:
	def minOperations(self, nums: List[int]) -> int:
		cnt = Counter(nums)
		total_ops = 0
		for count in cnt.values():
			if count == 1:
				return -1
			if count % 3 == 0:
				total_ops += count // 3
			else:
				total_ops += count // 3 + 1
		return total_ops