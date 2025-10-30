class Solution:
	def minimumLevels(self, possible: List[int]) -> int:
		total = sum(1 if x == 1 else -1 for x in possible)
		current = 0
		for i in range(len(possible) - 1):
			current += 1 if possible[i] == 1 else -1
			if 2 * current > total:
				return i + 1
		return -1