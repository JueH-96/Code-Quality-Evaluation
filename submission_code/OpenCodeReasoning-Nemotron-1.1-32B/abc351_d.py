import collections

def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	H, W = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(data[i].strip())
	
	dirs = [(1,0), (-1,0), (0,1), (0,-1)]
	
	dead_end = [[False] * W for _ in range(H)]
	for i in range(H):
		for j in range(W):
			if grid[i][j] == '.':
				for dx, dy in dirs:
					ni, nj = i + dx, j + dy
					if 0 <= ni < H and 0 <= nj < W:
						if grid[ni][nj] == '#':
							dead_end[i][j] = True
							break
	
	visited = [[False] * W for _ in range(H)]
	ans = 1
	
	for i in range(H):
		for j in range(W):
			if grid[i][j] == '.' and not dead_end[i][j] and not visited[i][j]:
				component = []
				queue = collections.deque()
				queue.append((i, j))
				visited[i][j] = True
				while queue:
					x, y = queue.popleft()
					component.append((x, y))
					for dx, dy in dirs:
						nx, ny = x + dx, y + dy
						if 0 <= nx < H and 0 <= ny < W:
							if grid[nx][ny] == '.' and not dead_end[nx][ny] and not visited[nx][ny]:
								visited[nx][ny] = True
								queue.append((nx, ny))
				
				dead_set = set()
				for (x, y) in component:
					for dx, dy in dirs:
						nx, ny = x + dx, y + dy
						if 0 <= nx < H and 0 <= ny < W:
							if grid[nx][ny] == '.' and dead_end[nx][ny]:
								dead_set.add((nx, ny))
				
				candidate = len(component) + len(dead_set)
				if candidate > ans:
					ans = candidate
					
	print(ans)

if __name__ == "__main__":
	main()