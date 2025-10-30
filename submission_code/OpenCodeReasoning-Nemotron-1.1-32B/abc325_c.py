import sys
from collections import deque

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	H, W = map(int, data[0].split())
	grid = data[1:1+H]
	
	visited = [[False] * W for _ in range(H)]
	count = 0
	for i in range(H):
		for j in range(W):
			if grid[i][j] == '#' and not visited[i][j]:
				count += 1
				queue = deque()
				queue.append((i, j))
				visited[i][j] = True
				while queue:
					x, y = queue.popleft()
					for dx in (-1, 0, 1):
						for dy in (-1, 0, 1):
							if dx == 0 and dy == 0:
								continue
							nx, ny = x + dx, y + dy
							if 0 <= nx < H and 0 <= ny < W:
								if grid[nx][ny] == '#' and not visited[nx][ny]:
									visited[nx][ny] = True
									queue.append((nx, ny))
	print(count)

if __name__ == "__main__":
	main()