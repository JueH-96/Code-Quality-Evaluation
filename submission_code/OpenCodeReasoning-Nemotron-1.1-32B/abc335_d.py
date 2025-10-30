n = int(input().strip())
center = (n - 1) // 2
grid = [['' for _ in range(n)] for __ in range(n)]
num = 1
for ring in range(0, center):
	for j in range(ring, n - ring):
		grid[ring][j] = str(num)
		num += 1
	for i in range(ring + 1, n - ring):
		grid[i][n - 1 - ring] = str(num)
		num += 1
	for j in range(n - 2 - ring, ring - 1, -1):
		grid[n - 1 - ring][j] = str(num)
		num += 1
	for i in range(n - 2 - ring, ring, -1):
		grid[i][ring] = str(num)
		num += 1
grid[center][center] = 'T'
for row in grid:
	print(" ".join(row))