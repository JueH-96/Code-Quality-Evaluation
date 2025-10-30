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
	
	x = X - 1
	y = Y - 1
	visited_houses = set()
	
	for move in T:
		nx, ny = x, y
		if move == 'U':
			nx = x - 1
		elif move == 'D':
			nx = x + 1
		elif move == 'L':
			ny = y - 1
		elif move == 'R':
			ny = y + 1
			
		if 0 <= nx < H and 0 <= ny < W:
			if grid[nx][ny] != '#':
				x, y = nx, ny
				
		if grid[x][y] == '@':
			visited_houses.add((x, y))
			
	print(f"{x+1} {y+1} {len(visited_houses)}")

if __name__ == "__main__":
	main()