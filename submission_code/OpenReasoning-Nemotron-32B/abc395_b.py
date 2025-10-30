def main():
	n = int(input().strip())
	grid = [['.' for _ in range(n)] for _ in range(n)]
	
	for i in range(1, n + 1):
		j = n + 1 - i
		if i <= j:
			color_char = '#' if i % 2 == 1 else '.'
			for r in range(i - 1, j):
				for c in range(i - 1, j):
					grid[r][c] = color_char
					
	for row in grid:
		print(''.join(row))

if __name__ == "__main__":
	main()