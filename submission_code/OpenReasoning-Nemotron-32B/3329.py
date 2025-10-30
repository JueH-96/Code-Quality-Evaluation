from typing import List

class Solution:
	def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
		prefix_set = set()
		for num in arr1:
			s = str(num)
			for i in range(1, len(s) + 1):
				prefix_set.add(s[:i])
		
		max_len = 0
		for num in arr2:
			s = str(num)
			current_len = 0
			for i in range(1, len(s) + 1):
				prefix = s[:i]
				if prefix in prefix_set:
					current_len = i
				else:
					break
			if current_len > max_len:
				max_len = current_len
		return max_len