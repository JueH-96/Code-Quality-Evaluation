from typing import List

class Solution:
	def countCompleteSubarrays(self, nums: List[int]) -> int:
		d = len(set(nums))
		last_occurrence = {}
		total = 0
		for i, num in enumerate(nums):
			last_occurrence[num] = i
			if len(last_occurrence) == d:
				min_index = min(last_occurrence.values())
				total += min_index + 1
		return total