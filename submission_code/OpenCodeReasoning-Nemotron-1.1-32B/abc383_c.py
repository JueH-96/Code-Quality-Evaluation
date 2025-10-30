import sys
from collections import deque

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		print(0)
		return
	H, W, D = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(list(data[i].strip()))
	
	dist = [[-1] * W for _ in range(H)]
	q = deque()
	for i in range(H):
		for j in range(W):
			if grid[i][j] == 'H':
				dist[i][j] = 0
				q.append((i, j))
	
	directions = [(1,0), (-1,0), (0,1), (0,-1)]
	while q:
		x, y = q.popleft()
		for dx, dy in directions:
			nx, ny = x + dx, y + dy
			if 0 <= nx < H and 0 <= ny < W:
				if grid[nx][ny] != '#' and dist[nx][ny] == -1:
					dist[nx][ny] = dist[x][y] + 1
					q.append((nx, ny))
	
	count = 0
	for i in range(H):
		for j in range(W):
			if grid[i][j] != '#' and dist[i][j] != -1 and dist[i][j] <= D:
				count += 1
	print(count)

if __name__ == '__main__':
	main()