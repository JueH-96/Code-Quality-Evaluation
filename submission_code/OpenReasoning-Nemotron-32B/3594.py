from collections import Counter

class Solution:
	def getLargestOutlier(self, nums: list) -> int:
		total = sum(nums)
		freq = Counter(nums)
		candidates = set()
		
		for x in nums:
			diff = total - x
			if diff % 2 != 0:
				continue
			T_val = diff // 2
			if x != T_val:
				if T_val in freq:
					candidates.add(x)
			else:
				if freq[x] >= 2:
					candidates.add(x)
					
		return max(candidates)