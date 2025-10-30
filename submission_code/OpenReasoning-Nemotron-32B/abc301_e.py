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
				
	if start is None or goal is None:
		print(-1)
		return
		
	points = [start, goal] + candies
	k = len(candies)
	n = len(points)
	
	INF = 10**9
	dist_matrix = [[INF] * n for _ in range(n)]
	directions = [(1,0), (-1,0), (0,1), (0,-1)]
	
	for idx in range(n):
		sx, sy = points[idx]
		dist_grid = [[INF] * W for _ in range(H)]
		q = deque()
		dist_grid[sx][sy] = 0
		q.append((sx, sy))
		while q:
			x, y = q.popleft()
			for dx, dy in directions:
				nx, ny = x + dx, y + dy
				if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
					if dist_grid[nx][ny] == INF:
						dist_grid[nx][ny] = dist_grid[x][y] + 1
						q.append((nx, ny))
		for j in range(n):
			xj, yj = points[j]
			dist_matrix[idx][j] = dist_grid[xj][yj]
			
	dp = [[INF] * n for _ in range(1<<k)]
	dp[0][0] = 0
	
	for mask in range(1<<k):
		for i in range(n):
			if dp[mask][i] == INF:
				continue
			for j in range(n):
				if i == j:
					continue
				d_ij = dist_matrix[i][j]
				if d_ij == INF:
					continue
				if j >= 2:
					candy_index = j - 2
					if mask & (1 << candy_index):
						new_mask = mask
					else:
						new_mask = mask | (1 << candy_index)
				else:
					new_mask = mask
					
				new_cost = dp[mask][i] + d_ij
				if new_cost < dp[new_mask][j]:
					dp[new_mask][j] = new_cost
					
	ans = -1
	for mask in range(1<<k):
		for i in range(n):
			if dp[mask][i] == INF:
				continue
			total_moves = dp[mask][i] + dist_matrix[i][1]
			if total_moves <= T:
				cnt = bin(mask).count("1")
				if cnt > ans:
					ans = cnt
					
	print(ans if ans != -1 else -1)

if __name__ == "__main__":
	main()