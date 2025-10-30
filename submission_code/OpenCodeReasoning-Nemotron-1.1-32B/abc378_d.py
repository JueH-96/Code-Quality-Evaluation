import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		print(0)
		return
	
	first_line = data[0].split()
	H = int(first_line[0])
	W = int(first_line[1])
	K = int(first_line[2])
	grid = []
	for i in range(1, 1+H):
		grid.append(data[i].strip())
	
	dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
	total = 0

	def dfs(i, j, depth, visited_grid):
		if depth == K:
			return 1
		count = 0
		for dx, dy in dirs:
			ni, nj = i + dx, j + dy
			if 0 <= ni < H and 0 <= nj < W:
				if grid[ni][nj] == '.' and not visited_grid[ni][nj]:
					visited_grid[ni][nj] = True
					count += dfs(ni, nj, depth + 1, visited_grid)
					visited_grid[ni][nj] = False
		return count

	for i in range(H):
		for j in range(W):
			if grid[i][j] == '.':
				visited_grid = [[False] * W for _ in range(H)]
				visited_grid[i][j] = True
				total += dfs(i, j, 0, visited_grid)
				
	print(total)

if __name__ == '__main__':
	main()