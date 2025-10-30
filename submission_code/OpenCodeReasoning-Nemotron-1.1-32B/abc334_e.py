mod = 998244353

import sys
from collections import deque

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
	base = 0
	dirs = [(0,1), (0,-1), (1,0), (-1,0)]
	
	comp_id = 0
	for i in range(H):
		for j in range(W):
			if grid[i][j] == '#' and comp[i][j] == -1:
				base += 1
				comp[i][j] = comp_id
				q = deque([(i, j)])
				while q:
					x, y = q.popleft()
					for dx, dy in dirs:
						nx, ny = x+dx, y+dy
						if 0 <= nx < H and 0 <= ny < W and grid[nx][ny]=='#' and comp[nx][ny]==-1:
							comp[nx][ny] = comp_id
							q.append((nx, ny))
				comp_id += 1
				
	total_red = 0
	total_k = 0
	for i in range(H):
		for j in range(W):
			if grid[i][j] == '.':
				total_red += 1
				seen = set()
				for dx, dy in dirs:
					ni, nj = i+dx, j+dy
					if 0<=ni<H and 0<=nj<W and grid[ni][nj]=='#':
						seen.add(comp[ni][nj])
				total_k += len(seen)
				
	numerator = (base + 1) * total_red - total_k
	numerator %= mod
	inv_total_red = pow(total_red, mod-2, mod)
	result = numerator * inv_total_red % mod
	print(result)

if __name__ == '__main__':
	main()