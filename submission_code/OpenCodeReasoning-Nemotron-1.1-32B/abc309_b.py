def main():
	n = int(input().strip())
	grid = [list(input().strip()) for _ in range(n)]
	
	outer = []
	for j in range(n):
		outer.append(grid[0][j])
		
	for i in range(1, n-1):
		outer.append(grid[i][n-1])
		
	for j in range(n-1, -1, -1):
		outer.append(grid[n-1][j])
		
	for i in range(n-2, 0, -1):
		outer.append(grid[i][0])
		
	if outer:
		shifted_outer = [outer[-1]] + outer[:-1]
	else:
		shifted_outer = outer
		
	idx = 0
	for j in range(n):
		grid[0][j] = shifted_outer[idx]
		idx += 1
		
	for i in range(1, n-1):
		grid[i][n-1] = shifted_outer[idx]
		idx += 1
		
	for j in range(n-1, -1, -1):
		grid[n-1][j] = shifted_outer[idx]
		idx += 1
		
	for i in range(n-2, 0, -1):
		grid[i][0] = shifted_outer[idx]
		idx += 1
		
	for row in grid:
		print(''.join(row))

if __name__ == '__main__':
	main()