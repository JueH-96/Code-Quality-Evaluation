import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	try:
		N = int(data[0].strip())
	except:
		return
	
	if N == 0:
		print("#")
		return
		
	grid = ["#"]
	for level in range(1, N+1):
		size_prev = len(grid)
		new_grid = []
		for i in range(size_prev):
			new_grid.append(grid[i] * 3)
		for i in range(size_prev):
			new_grid.append(grid[i] + ('.' * size_prev) + grid[i])
		for i in range(size_prev):
			new_grid.append(grid[i] * 3)
		grid = new_grid
		
	for row in grid:
		print(row)

if __name__ == '__main__':
	main()