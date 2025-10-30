class Solution:
	def findMissingAndRepeatedValues(self, grid: List[List[int]]) -> List[int]:
		n = len(grid)
		total_squares = n * n
		freq = [0] * (total_squares + 1)
		for row in grid:
			for num in row:
				freq[num] += 1
		duplicate = None
		missing = None
		for i in range(1, total_squares + 1):
			if freq[i] == 0:
				missing = i
			elif freq[i] == 2:
				duplicate = i
		return [duplicate, missing]