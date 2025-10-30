def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	
	H, W, X, Y = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(data[i].strip())
	T = data[1+H].strip()
	
	r, c = X-1, Y-1
	houses = set()
	
	for move in T:
		dr, dc = 0, 0
		if move == 'U':
			dr = -1
		elif move == 'D':
			dr = 1
		elif move == 'L':
			dc = -1
		elif move == 'R':
			dc = 1
			
		nr = r + dr
		nc = c + dc
		if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
			r, c = nr, nc
			
		if grid[r][c] == '@':
			houses.add((r, c))
			
	print(f"{r+1} {c+1} {len(houses)}")

if __name__ == "__main__":
	main()