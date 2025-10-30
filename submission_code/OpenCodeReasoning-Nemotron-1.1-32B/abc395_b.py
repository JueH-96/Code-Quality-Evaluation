def main():
	n = int(input().strip())
	if n == 0:
		return
	half = (n - 1) // 2
	grid = []
	for r in range(n):
		row_chars = []
		for c in range(n):
			a = min(r, c)
			b = max(r, c)
			candidate = min(a, n - 1 - b)
			i0_max = min(candidate, half)
			if i0_max % 2 == 0:
				row_chars.append('#')
			else:
				row_chars.append('.')
		grid.append(''.join(row_chars))
	
	for row in grid:
		print(row)

if __name__ == '__main__':
	main()