import itertools

class Solution:
	def findMinimumTime(self, strength: List[int], K: int) -> int:
		n = len(strength)
		min_time = float('inf')
		for perm in itertools.permutations(strength):
			total = 0
			for j in range(n):
				divisor = 1 + j * K
				t_j = (perm[j] + divisor - 1) // divisor
				total += t_j
			if total < min_time:
				min_time = total
		return min_time