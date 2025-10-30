def main():
	grid = []
	for _ in range(8):
		grid.append(input().strip())
	
	rows_with_piece = set()
	cols_with_piece = set()
	
	for i in range(8):
		for j in range(8):
			if grid[i][j] == '#':
				rows_with_piece.add(i)
				cols_with_piece.add(j)
				
	count = 0
	for i in range(8):
		for j in range(8):
			if grid[i][j] == '.':
				if i not in rows_with_piece and j not in cols_with_piece:
					count += 1
					
	print(count)

if __name__ == "__main__":
	main()