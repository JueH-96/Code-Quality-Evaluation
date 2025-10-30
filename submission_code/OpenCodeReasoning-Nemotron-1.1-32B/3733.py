class Solution:
	def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
		n = len(grid)
		m = len(grid[0])
		dirs = [(1,1), (1,-1), (-1,-1), (-1,1)]
		
		if n == 0 or m == 0:
			return 0
		
		dp = [[[ [0, 0] for _ in range(4) ] for _ in range(m)] for _ in range(n)]
		
		for d in range(4):
			dr, dc = dirs[d]
			if dr == 1 and dc == 1:
				for i in range(n-1, -1, -1):
					for j in range(m-1, -1, -1):
						if grid[i][j] == 0:
							ni, nj = i + dr, j + dc
							if 0 <= ni < n and 0 <= nj < m:
								dp[i][j][d][0] = 1 + dp[ni][nj][d][1]
							else:
								dp[i][j][d][0] = 1
						else:
							dp[i][j][d][0] = 0
							
						if grid[i][j] == 2:
							ni, nj = i + dr, j + dc
							if 0 <= ni < n and 0 <= nj < m:
								dp[i][j][d][1] = 1 + dp[ni][nj][d][0]
							else:
								dp[i][j][d][1] = 1
						else:
							dp[i][j][d][1] = 0
			elif dr == 1 and dc == -1:
				for i in range(n-1, -1, -1):
					for j in range(0, m):
						if grid[i][j] == 0:
							ni, nj = i + dr, j + dc
							if 0 <= ni < n and 0 <= nj < m:
								dp[i][j][d][0] = 1 + dp[ni][nj][d][1]
							else:
								dp[i][j][d][0] = 1
						else:
							dp[i][j][d][0] = 0
							
						if grid[i][j] == 2:
							ni, nj = i + dr, j + dc
							if 0 <= ni < n and 0 <= nj < m:
								dp[i][j][d][1] = 1 + dp[ni][nj][d][0]
							else:
								dp[i][j][d][1] = 1
						else:
							dp[i][j][d][1] = 0
			elif dr == -1 and dc == -1:
				for i in range(0, n):
					for j in range(0, m):
						if grid[i][j] == 0:
							ni, nj = i + dr, j + dc
							if 0 <= ni < n and 0 <= nj < m:
								dp[i][j][d][0] = 1 + dp[ni][nj][d][1]
							else:
								dp[i][j][d][0] = 1
						else:
							dp[i][j][d][0] = 0
							
						if grid[i][j] == 2:
							ni, nj = i + dr, j + dc
							if 0 <= ni < n and 0 <= nj < m:
								dp[i][j][d][1] = 1 + dp[ni][nj][d][0]
							else:
								dp[i][j][d][1] = 1
						else:
							dp[i][j][d][1] = 0
			elif dr == -1 and dc == 1:
				for i in range(0, n):
					for j in range(m-1, -1, -1):
						if grid[i][j] == 0:
							ni, nj = i + dr, j + dc
							if 0 <= ni < n and 0 <= nj < m:
								dp[i][j][d][0] = 1 + dp[ni][nj][d][1]
							else:
								dp[i][j][d][0] = 1
						else:
							dp[i][j][d][0] = 0
							
						if grid[i][j] == 2:
							ni, nj = i + dr, j + dc
							if 0 <= ni < n and 0 <= nj < m:
								dp[i][j][d][1] = 1 + dp[ni][nj][d][0]
							else:
								dp[i][j][d][1] = 1
						else:
							dp[i][j][d][1] = 0
		
		ans = 0
		for i in range(n):
			for j in range(m):
				if grid[i][j] == 1:
					for d in range(4):
						dr, dc = dirs[d]
						cur_r, cur_c = i, j
						step_index = 0
						best_here = 1
						
						while True:
							nr = cur_r + dr
							nc = cur_c + dc
							if not (0 <= nr < n and 0 <= nc < m):
								break
							required_val = 2 if (step_index + 1) % 2 == 1 else 0
							if grid[nr][nc] != required_val:
								break
							
							cur_r, cur_c = nr, nc
							step_index += 1
							
							if step_index + 1 > best_here:
								best_here = step_index + 1
							
							d_next = (d + 1) % 4
							dr2, dc2 = dirs[d_next]
							p = (step_index + 1) % 2
							next_r = cur_r + dr2
							next_c = cur_c + dc2
							if 0 <= next_r < n and 0 <= next_c < m:
								count_second = dp[next_r][next_c][d_next][p]
							else:
								count_second = 0
							
							total_length = step_index + 1 + count_second
							if total_length > best_here:
								best_here = total_length
						
						if best_here > ans:
							ans = best_here
		return ans