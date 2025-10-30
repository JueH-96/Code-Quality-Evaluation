class Solution:
	def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
		n = len(grid)
		m = len(grid[0])
		directions = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
		dp = [[[1] * 4 for _ in range(m)] for __ in range(n)]
		
		def next_val(x):
			if x == 2:
				return 0
			else:
				return 2
		
		for d in range(4):
			dx, dy = directions[d]
			if dx == 1:
				irange = range(n-1, -1, -1)
			else:
				irange = range(0, n)
			if dy == 1:
				jrange = range(m-1, -1, -1)
			else:
				jrange = range(0, m)
				
			for i in irange:
				for j in jrange:
					ni = i + dx
					nj = j + dy
					if 0 <= ni < n and 0 <= nj < m:
						if grid[ni][nj] == next_val(grid[i][j]):
							dp[i][j][d] = dp[ni][nj][d] + 1
		
		ans = 0
		for i in range(n):
			for j in range(m):
				if grid[i][j] == 1:
					for d in range(4):
						if dp[i][j][d] > ans:
							ans = dp[i][j][d]
		
		for i in range(n):
			for j in range(m):
				if grid[i][j] == 1:
					for d1 in range(4):
						L1 = dp[i][j][d1]
						dx1, dy1 = directions[d1]
						for k in range(1, L1):
							i1 = i + k * dx1
							j1 = j + k * dy1
							d2 = (d1 + 1) % 4
							L2 = dp[i1][j1][d2]
							total_length = k + L2
							if total_length > ans:
								ans = total_length
		return ans