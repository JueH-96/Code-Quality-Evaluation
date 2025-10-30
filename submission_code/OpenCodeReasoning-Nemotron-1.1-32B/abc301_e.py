import sys
from collections import deque

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(-1)
		return
	
	H, W, T = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(data[i].strip())
	
	start = None
	goal = None
	candies = []
	for i in range(H):
		for j in range(W):
			if grid[i][j] == 'S':
				start = (i, j)
			elif grid[i][j] == 'G':
				goal = (i, j)
			elif grid[i][j] == 'o':
				candies.append((i, j))
				
	k = len(candies)
	points = [start] + candies + [goal]
	n_points = len(points)
	
	dirs = [(0,1), (1,0), (0,-1), (-1,0)]
	
	def bfs(grid, start):
		H = len(grid)
		W = len(grid[0])
		dist = [[10**9] * W for _ in range(H)]
		sx, sy = start
		dist[sx][sy] = 0
		q = deque()
		q.append((sx, sy))
		while q:
			x, y = q.popleft()
			for dx, dy in dirs:
				nx, ny = x + dx, y + dy
				if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
					if dist[nx][ny] > dist[x][y] + 1:
						dist[nx][ny] = dist[x][y] + 1
						q.append((nx, ny))
		return dist
	
	dist_mat = [[0] * n_points for _ in range(n_points)]
	for idx in range(n_points):
		d_grid = bfs(grid, points[idx])
		for j in range(n_points):
			x, y = points[j]
			dist_mat[idx][j] = d_grid[x][y]
			
	if dist_mat[0][n_points-1] > T or dist_mat[0][n_points-1] == 10**9:
		print(-1)
		return
		
	dp = [[10**9] * (k+1) for _ in range(1<<k)]
	dp[0][0] = 0
	ans = -1
	
	for mask in range(1<<k):
		for i in range(0, k+1):
			if dp[mask][i] > T:
				continue
			total_moves = dp[mask][i] + dist_mat[i][n_points-1]
			if total_moves <= T:
				cnt = bin(mask).count("1")
				if cnt > ans:
					ans = cnt
			cur = dp[mask][i]
			for j in range(1, k+1):
				if mask & (1 << (j-1)):
					continue
				new_mask = mask | (1 << (j-1))
				new_cost = cur + dist_mat[i][j]
				if new_cost < dp[new_mask][j]:
					dp[new_mask][j] = new_cost
					
	print(ans)

if __name__ == "__main__":
	main()