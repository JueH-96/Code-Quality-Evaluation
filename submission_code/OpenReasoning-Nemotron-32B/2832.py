from collections import defaultdict

class Solution:
	def longestEqualSubarray(self, nums: list[int], k: int) -> int:
		indices_map = defaultdict(list)
		for i, num in enumerate(nums):
			indices_map[num].append(i)
		
		ans = 0
		for indices in indices_map.values():
			left = 0
			for right in range(len(indices)):
				while indices[right] - indices[left] - (right - left) > k:
					left += 1
				ans = max(ans, right - left + 1)
		return ans