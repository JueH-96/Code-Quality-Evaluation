from collections import deque
import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
		
	first_line = data[0].split()
	H = int(first_line[0])
	W = int(first_line[1])
	D = int(first_line[2])
	grid = []
	for i in range(1, 1+H):
		grid.append(list(data[i].strip()))
	
	dist = [[-1] * W for _ in range(H)]
	q = deque()
	directions = [(1,0), (-1,0), (0,1), (0,-1)]
	
	for i in range(H):
		for j in range(W):
			if grid[i][j] == 'H':
				dist[i][j] = 0
				q.append((i, j))
				
	while q:
		i, j = q.popleft()
		if dist[i][j] < D:
			for dx, dy in directions:
				ni, nj = i + dx, j + dy
				if 0 <= ni < H and 0 <= nj < W:
					if grid[ni][nj] != '#' and dist[ni][nj] == -1:
						dist[ni][nj] = dist[i][j] + 1
						q.append((ni, nj))
						
	count = 0
	for i in range(H):
		for j in range(W):
			if grid[i][j] != '#' and dist[i][j] != -1:
				count += 1
				
	print(count)

if __name__ == "__main__":
	main()