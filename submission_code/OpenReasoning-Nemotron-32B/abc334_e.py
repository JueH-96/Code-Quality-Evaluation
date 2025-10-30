import sys
from collections import deque

MOD = 998244353

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		print(0)
		return
	H, W = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(data[i].strip())
	
	comp = [[-1] * W for _ in range(H)]
	comp_id = 0
	directions = [(1,0), (-1,0), (0,1), (0,-1)]
	
	for i in range(H):
		for j in range(W):
			if grid[i][j] == '#' and comp[i][j] == -1:
				comp_id += 1
				q = deque()
				q.append((i, j))
				comp[i][j] = comp_id
				while q:
					x, y = q.popleft()
					for dx, dy in directions:
						nx, ny = x + dx, y + dy
						if 0 <= nx < H and 0 <= ny < W and grid[nx][ny]=='#' and comp[nx][ny] == -1:
							comp[nx][ny] = comp_id
							q.append((nx, ny))
							
	C0 = comp_id
	
	total_red = 0
	for i in range(H):
		for j in range(W):
			if grid[i][j] == '.':
				total_red += 1
				
	S = 0
	for i in range(H):
		for j in range(W):
			if grid[i][j] == '.':
				comp_set = set()
				for dx, dy in directions:
					ni, nj = i + dx, j + dy
					if 0 <= ni < H and 0 <= nj < W and grid[ni][nj]=='#':
						comp_set.add(comp[ni][nj])
				S += len(comp_set)
				
	A = (C0 * total_red + total_red - S) % MOD
	if total_red == 0:
		R = 0
	else:
		inv_total_red = pow(total_red, MOD-2, MOD)
		R = (A * inv_total_red) % MOD
		
	print(R)

if __name__ == '__main__':
	main()