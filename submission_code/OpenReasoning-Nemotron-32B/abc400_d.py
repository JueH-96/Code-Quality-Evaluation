from collections import deque
import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	
	H, W = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(list(data[i].strip()))
	
	A, B, C, D = map(int, data[1+H].split())
	A, B, C, D = A-1, B-1, C-1, D-1

	if A == C and B == D:
		print(0)
		return

	INF = 10**9
	dist = [[INF] * W for _ in range(H)]
	dist[A][B] = 0
	q = deque()
	q.append((0, A, B))
	
	while q:
		k, r, c = q.popleft()
		
		if k != dist[r][c]:
			continue
			
		if r == C and c == D:
			print(k)
			return
			
		for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
			nr = r + dr
			nc = c + dc
			if 0 <= nr < H and 0 <= nc < W:
				if grid[nr][nc] == '.':
					if k < dist[nr][nc]:
						dist[nr][nc] = k
						q.appendleft((k, nr, nc))
						
		for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
			nr1 = r + dr
			nc1 = c + dc
			broken_step1 = False
			if 0 <= nr1 < H and 0 <= nc1 < W:
				if grid[nr1][nc1] == '#':
					grid[nr1][nc1] = '.'
					broken_step1 = True
					
			nr2 = r + 2*dr
			nc2 = c + 2*dc
			if 0 <= nr2 < H and 0 <= nc2 < W:
				if grid[nr2][nc2] == '#':
					grid[nr2][nc2] = '.'
					
			if broken_step1:
				if k+1 < dist[nr1][nc1]:
					dist[nr1][nc1] = k+1
					q.append((k+1, nr1, nc1))
					
	if dist[C][D] == INF:
		print(-1)
	else:
		print(dist[C][D])

if __name__ == "__main__":
	main()