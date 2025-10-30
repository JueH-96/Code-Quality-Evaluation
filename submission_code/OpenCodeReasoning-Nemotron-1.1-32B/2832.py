from collections import defaultdict
from typing import List

class Solution:
	def longestEqualSubarray(self, nums: List[int], k: int) -> int:
		group = defaultdict(list)
		for idx, num in enumerate(nums):
			group[num].append(idx)
		
		ans = 0
		for key, indices in group.items():
			left = 0
			n = len(indices)
			for right in range(n):
				while indices[right] - indices[left] - (right - left) > k:
					left += 1
				current_length = right - left + 1
				if current_length > ans:
					ans = current_length
		return ans