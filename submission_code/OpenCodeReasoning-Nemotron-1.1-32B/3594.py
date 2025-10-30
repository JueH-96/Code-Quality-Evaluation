from collections import Counter
from typing import List

class Solution:
	def getLargestOutlier(self, nums: List[int]) -> int:
		total = sum(nums)
		freq = Counter(nums)
		candidates = set()
		for num in freq:
			X = total - 2 * num
			if X in freq:
				if num == X:
					if freq[num] >= 2:
						candidates.add(X)
				else:
					candidates.add(X)
		return max(candidates)