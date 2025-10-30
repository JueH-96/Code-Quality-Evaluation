import itertools
from typing import List

class Solution:
	def findMinimumTime(self, strength: List[int], K: int) -> int:
		min_time = float('inf')
		for perm in itertools.permutations(strength):
			total_time = 0
			X = 1
			for s in perm:
				t = (s + X - 1) // X
				total_time += t
				X += K
			if total_time < min_time:
				min_time = total_time
		return min_time