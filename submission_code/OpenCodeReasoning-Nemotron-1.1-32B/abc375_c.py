import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n = int(data[0].strip())
	grid = []
	for i in range(1, 1+n):
		grid.append(list(data[i].strip()))
	
	for i in range(1, n//2+1):
		new_grid = [row[:] for row in grid]
		L = i-1
		R = n - i
		for x0 in range(L, R+1):
			for y0 in range(L, R+1):
				new_grid[y0][n-1-x0] = grid[x0][y0]
		grid = new_grid

	for row in grid:
		print(''.join(row))

if __name__ == "__main__":
	main()