def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	H, W, D = map(int, data[0].split())
	grid = []
	for i in range(1, 1 + H):
		grid.append(data[i].strip())
	
	floor_cells = []
	for i in range(H):
		for j in range(W):
			if grid[i][j] == '.':
				floor_cells.append((i, j))
				
	n = len(floor_cells)
	max_covered = 0
	
	for i in range(n):
		for j in range(i + 1, n):
			count = 0
			A = floor_cells[i]
			B = floor_cells[j]
			for k in range(n):
				cell = floor_cells[k]
				d1 = abs(A[0] - cell[0]) + abs(A[1] - cell[1])
				d2 = abs(B[0] - cell[0]) + abs(B[1] - cell[1])
				if d1 <= D or d2 <= D:
					count += 1
			if count > max_covered:
				max_covered = count
				
	print(max_covered)

if __name__ == '__main__':
	main()