class Solution:
	def zigzagTraversal(self, grid: List[List[int]]) -> List[int]:
		n = len(grid)
		m = len(grid[0])
		take_next = True
		res = []
		for i in range(n):
			if i % 2 == 0:
				for j in range(m):
					if take_next:
						res.append(grid[i][j])
					take_next = not take_next
			else:
				for j in range(m-1, -1, -1):
					if take_next:
						res.append(grid[i][j])
					take_next = not take_next
		return res