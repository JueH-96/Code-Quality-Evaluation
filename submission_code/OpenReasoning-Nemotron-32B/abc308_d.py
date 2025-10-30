import collections

def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		print("No")
		return
	H, W = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(data[i].strip())
	
	pattern = "snuke"
	
	if grid[0][0] != pattern[0]:
		print("No")
		return
		
	visited = [[[False] * 5 for _ in range(W)] for __ in range(H)]
	visited[0][0][0] = True
	
	directions = [(1,0), (-1,0), (0,1), (0,-1)]
	
	from collections import deque
	q = deque()
	q.append((0,0,0))
	
	while q:
		i, j, phase = q.popleft()
		if i == H-1 and j == W-1:
			print("Yes")
			return
			
		next_phase = (phase + 1) % 5
		for dx, dy in directions:
			ni, nj = i + dx, j + dy
			if 0 <= ni < H and 0 <= nj < W:
				if grid[ni][nj] == pattern[next_phase]:
					if not visited[ni][nj][next_phase]:
						visited[ni][nj][next_phase] = True
						q.append((ni, nj, next_phase))
						
	print("No")

if __name__ == "__main__":
	main()