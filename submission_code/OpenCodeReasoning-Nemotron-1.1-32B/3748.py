from collections import defaultdict
from typing import List

class Solution:
	def sortMatrix(self, grid: List[List[int]]) -> List[List[int]]:
		n = len(grid)
		groups = defaultdict(list)
		for i in range(n):
			for j in range(n):
				d = i - j
				groups[d].append((i, j))
		
		for d, coords in groups.items():
			values = [grid[i][j] for i, j in coords]
			if d >= 0:
				values.sort(reverse=True)
			else:
				values.sort()
			for idx, (i, j) in enumerate(coords):
				grid[i][j] = values[idx]
		
		return grid