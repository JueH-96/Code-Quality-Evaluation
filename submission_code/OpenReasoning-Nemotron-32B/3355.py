class Solution:
	def minimumLevels(self, possible: List[int]) -> int:
		n = len(possible)
		total_ones = sum(possible)
		current_ones = 0
		for i in range(n - 1):
			current_ones += possible[i]
			if 4 * current_ones > 2 * total_ones - n + 2 * (i + 1):
				return i + 1
		return -1