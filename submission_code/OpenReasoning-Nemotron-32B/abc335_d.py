def main():
	N = int(input().strip())
	c = (N - 1) // 2
	grid = [[''] * N for _ in range(N)]
	grid[c][c] = 'T'
	
	total = N * N - 1
	count = 1
	top, bottom, left, right = 0, N - 1, 0, N - 1
	d = 0
	
	while count <= total:
		if d == 0:
			for j in range(left, right + 1):
				if top == c and j == c:
					continue
				grid[top][j] = str(count)
				count += 1
			top += 1
		elif d == 1:
			for i in range(top, bottom + 1):
				if i == c and right == c:
					continue
				grid[i][right] = str(count)
				count += 1
			right -= 1
		elif d == 2:
			for j in range(right, left - 1, -1):
				if bottom == c and j == c:
					continue
				grid[bottom][j] = str(count)
				count += 1
			bottom -= 1
		elif d == 3:
			for i in range(bottom, top - 1, -1):
				if i == c and left == c:
					continue
				grid[i][left] = str(count)
				count += 1
			left += 1
		d = (d + 1) % 4
	
	for row in grid:
		print(' '.join(row))

if __name__ == '__main__':
	main()