import bisect
from typing import List

class Solution:
	def solveQueries(self, nums: List[int], queries: List[int]) -> List[int]:
		n = len(nums)
		index_map = {}
		for i, num in enumerate(nums):
			index_map.setdefault(num, []).append(i)
		for key in index_map:
			index_map[key].sort()
		res = []
		for q in queries:
			v = nums[q]
			lst = index_map[v]
			if len(lst) == 1:
				res.append(-1)
			else:
				pos = bisect.bisect_left(lst, q)
				candidates = set()
				if pos > 0:
					candidates.add(lst[pos-1])
				if pos < len(lst)-1:
					candidates.add(lst[pos+1])
				candidates.add(lst[0])
				candidates.add(lst[-1])
				candidates.discard(q)
				min_dist = float('inf')
				for j in candidates:
					d = abs(q - j)
					circular_d = min(d, n - d)
					if circular_d < min_dist:
						min_dist = circular_d
				res.append(min_dist)
		return res