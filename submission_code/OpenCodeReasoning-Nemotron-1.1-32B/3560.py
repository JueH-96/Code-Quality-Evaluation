from collections import deque
from functools import lru_cache

class Solution:
	def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
		n = len(positions)
		points = [(kx, ky)] + positions
		
		if n == 0:
			return 0
		
		moves = [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]
		D = [[0] * (n+1) for _ in range(n+1)]
		
		for i in range(n+1):
			sx, sy = points[i]
			dist_grid = [[-1] * 50 for _ in range(50)]
			q = deque()
			dist_grid[sx][sy] = 0
			q.append((sx, sy))
			while q:
				x, y = q.popleft()
				for dx, dy in moves:
					nx, ny = x + dx, y + dy
					if 0 <= nx < 50 and 0 <= ny < 50 and dist_grid[nx][ny] == -1:
						dist_grid[nx][ny] = dist_grid[x][y] + 1
						q.append((nx, ny))
			for j in range(n+1):
				xj, yj = points[j]
				D[i][j] = dist_grid[xj][yj]
				
		@lru_cache(maxsize=None)
		def dp(mask, pos):
			if mask == 0:
				return 0
			n_remaining = bin(mask).count("1")
			if (n - n_remaining) % 2 == 0:
				res = -10**9
				for j in range(n):
					if mask & (1 << j):
						new_mask = mask & ~(1 << j)
						moves_here = D[pos][j+1]
						total_moves = moves_here + dp(new_mask, j+1)
						if total_moves > res:
							res = total_moves
				return res
			else:
				res = 10**9
				for j in range(n):
					if mask & (1 << j):
						new_mask = mask & ~(1 << j)
						moves_here = D[pos][j+1]
						total_moves = moves_here + dp(new_mask, j+1)
						if total_moves < res:
							res = total_moves
				return res
		
		return dp((1 << n) - 1, 0)