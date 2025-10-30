def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n = int(data[0].strip())
	grid = []
	for i in range(1, 1 + n):
		grid.append(list(data[i].strip()))
	
	ring = []
	for j in range(n):
		ring.append(grid[0][j])
	
	for i in range(1, n - 1):
		ring.append(grid[i][n - 1])
	
	for j in range(n - 1, -1, -1):
		ring.append(grid[n - 1][j])
	
	for i in range(n - 2, 0, -1):
		ring.append(grid[i][0])
	
	new_ring = [ring[-1]] + ring[:-1]
	
	index = 0
	for j in range(n):
		grid[0][j] = new_ring[index]
		index += 1
	
	for i in range(1, n - 1):
		grid[i][n - 1] = new_ring[index]
		index += 1
	
	for j in range(n - 1, -1, -1):
		grid[n - 1][j] = new_ring[index]
		index += 1
	
	for i in range(n - 2, 0, -1):
		grid[i][0] = new_ring[index]
		index += 1
	
	for row in grid:
		print(''.join(row))

if __name__ == "__main__":
	main()