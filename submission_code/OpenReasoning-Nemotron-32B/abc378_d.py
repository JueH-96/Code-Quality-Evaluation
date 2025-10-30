def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	H, W, K = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(data[i].strip())
	
	directions = [(0,1), (1,0), (0,-1), (-1,0)]
	total = 0
	
	for i in range(H):
		for j in range(W):
			if grid[i][j] == '.':
				visited = [[False] * W for _ in range(H)]
				stack = []
				visited[i][j] = True
				stack.append((i, j, 0, 0))
				count_start = 0
				while stack:
					i1, j1, depth, d = stack.pop()
					if depth == K:
						count_start += 1
						visited[i1][j1] = False
						continue
					
					if d < 4:
						stack.append((i1, j1, depth, d+1))
						dx, dy = directions[d]
						ni, nj = i1 + dx, j1 + dy
						if 0 <= ni < H and 0 <= nj < W:
							if grid[ni][nj] == '.' and not visited[ni][nj]:
								visited[ni][nj] = True
								stack.append((ni, nj, depth+1, 0))
					else:
						visited[i1][j1] = False
				total += count_start
	print(total)

if __name__ == "__main__":
	main()