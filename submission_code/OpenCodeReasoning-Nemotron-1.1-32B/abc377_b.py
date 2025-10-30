def main():
	grid = []
	for _ in range(8):
		grid.append(input().strip())
	
	rows_with_pieces = set()
	cols_with_pieces = set()
	
	for i in range(8):
		for j in range(8):
			if grid[i][j] == '#':
				rows_with_pieces.add(i)
				cols_with_pieces.add(j)
				
	count = 0
	for i in range(8):
		for j in range(8):
			if grid[i][j] == '.':
				if i not in rows_with_pieces and j not in cols_with_pieces:
					count += 1
					
	print(count)

if __name__ == "__main__":
	main()