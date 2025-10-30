class Solution:
	def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
		total_index = 0
		result = []
		m = len(grid)
		n = len(grid[0])
		for i in range(m):
			if i % 2 == 0:
				for j in range(n):
					if total_index % 2 == 0:
						result.append(grid[i][j])
					total_index += 1
			else:
				for j in range(n-1, -1, -1):
					if total_index % 2 == 0:
						result.append(grid[i][j])
					total_index += 1
		return result