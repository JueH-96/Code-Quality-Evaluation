n = int(input().strip())
grid = [['#']]
size = 1
for k in range(1, n+1):
	new_size = size * 3
	new_grid = [['.'] * new_size for _ in range(new_size)]
	for i in range(new_size):
		for j in range(new_size):
			block_row = i // size
			block_col = j // size
			if block_row != 1 or block_col != 1:
				new_grid[i][j] = grid[i % size][j % size]
	grid = new_grid
	size = new_size

for row in grid:
	print(''.join(row))