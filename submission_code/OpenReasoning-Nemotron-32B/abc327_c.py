grid = []
for _ in range(9):
	row = list(map(int, input().split()))
	grid.append(row)

target = set(range(1, 10))

# Check each row
for i in range(9):
	if set(grid[i]) != target:
		print("No")
		exit(0)

# Check each column
for j in range(9):
	col = [grid[i][j] for i in range(9)]
	if set(col) != target:
		print("No")
		exit(0)

# Check each 3x3 block
for i in range(0, 9, 3):
	for j in range(0, 9, 3):
		block = [grid[x][y] for x in range(i, i+3) for y in range(j, j+3)]
		if set(block) != target:
			print("No")
			exit(0)

print("Yes")