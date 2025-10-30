grid = []
for _ in range(9):
	row = list(map(int, input().split()))
	grid.append(row)

for i in range(9):
	if len(set(grid[i])) != 9:
		print("No")
		exit(0)

for j in range(9):
	col = [grid[i][j] for i in range(9)]
	if len(set(col)) != 9:
		print("No")
		exit(0)

for i in range(0, 9, 3):
	for j in range(0, 9, 3):
		block = []
		for x in range(i, i + 3):
			for y in range(j, j + 3):
				block.append(grid[x][y])
		if len(set(block)) != 9:
			print("No")
			exit(0)

print("Yes")