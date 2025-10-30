from collections import deque
from functools import lru_cache

class Solution:
	def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
		n = len(positions)
		points = [(kx, ky)] + positions
		
		moves = [(2,1), (1,2), (-1,2), (-2,1), (-2,-1), (-1,-2), (1,-2), (2,-1)]
		dists = [[0] * (n+1) for _ in range(n+1)]
		
		for i in range(n+1):
			sx, sy = points[i]
			dist_board = [[-1] * 50 for _ in range(50)]
			q = deque()
			dist_board[sx][sy] = 0
			q.append((sx, sy))
			while q:
				x, y = q.popleft()
				for dx, dy in moves:
					nx, ny = x + dx, y + dy
					if 0 <= nx < 50 and 0 <= ny < 50 and dist_board[nx][ny] == -1:
						dist_board[nx][ny] = dist_board[x][y] + 1
						q.append((nx, ny))
			for j in range(n+1):
				xj, yj = points[j]
				dists[i][j] = dist_board[xj][yj]
		
		@lru_cache(maxsize=None)
		def dp(mask, pos):
			if mask == 0:
				return 0
			k = bin(mask).count("1")
			turn = (n - k) % 2
			res = None
			for j in range(n):
				if mask & (1 << j):
					cost = dists[pos][j+1]
					new_mask = mask ^ (1 << j)
					new_pos = j + 1
					next_val = dp(new_mask, new_pos)
					total = cost + next_val
					if res is None:
						res = total
					else:
						if turn == 0:
							if total > res:
								res = total
						else:
							if total < res:
								res = total
			return res
		
		return dp((1 << n) - 1, 0)