import sys
from collections import deque

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(1)
		return
	H, W = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(data[i].strip())
	
	if H == 0 or W == 0:
		print(1)
		return
		
	safe = [[False] * W for _ in range(H)]
	dirs = [(1,0), (-1,0), (0,1), (0,-1)]
	
	for i in range(H):
		for j in range(W):
			if grid[i][j] == '#':
				continue
			has_magnet_adjacent = False
			for dx, dy in dirs:
				ni, nj = i + dx, j + dy
				if 0 <= ni < H and 0 <= nj < W:
					if grid[ni][nj] == '#':
						has_magnet_adjacent = True
						break
			safe[i][j] = not has_magnet_adjacent

	visited = [[False] * W for _ in range(H)]
	max_degree = 1
	
	for i in range(H):
		for j in range(W):
			if grid[i][j] == '.' and safe[i][j] and not visited[i][j]:
				queue = deque()
				queue.append((i, j))
				visited[i][j] = True
				count_safe = 0
				non_safe_adjacent = set()
				
				while queue:
					x, y = queue.popleft()
					count_safe += 1
					for dx, dy in dirs:
						nx, ny = x + dx, y + dy
						if 0 <= nx < H and 0 <= ny < W:
							if grid[nx][ny] == '.':
								if safe[nx][ny]:
									if not visited[nx][ny]:
										visited[nx][ny] = True
										queue.append((nx, ny))
								else:
									non_safe_adjacent.add((nx, ny))
				
				total = count_safe + len(non_safe_adjacent)
				if total > max_degree:
					max_degree = total
					
	print(max_degree)

if __name__ == '__main__':
	main()