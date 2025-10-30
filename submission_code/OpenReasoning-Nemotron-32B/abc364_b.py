def main():
	import sys
	data = sys.stdin.read().splitlines()
	H, W = map(int, data[0].split())
	start_line = data[1].split()
	s_i = int(start_line[0]) - 1
	s_j = int(start_line[1]) - 1
	
	grid = []
	for i in range(2, 2 + H):
		grid.append(data[i].strip())
	
	X = data[2 + H].strip()
	
	current_row = s_i
	current_col = s_j
	
	for move in X:
		if move == 'L':
			if current_col - 1 >= 0 and grid[current_row][current_col - 1] == '.':
				current_col -= 1
		elif move == 'R':
			if current_col + 1 < W and grid[current_row][current_col + 1] == '.':
				current_col += 1
		elif move == 'U':
			if current_row - 1 >= 0 and grid[current_row - 1][current_col] == '.':
				current_row -= 1
		elif move == 'D':
			if current_row + 1 < H and grid[current_row + 1][current_col] == '.':
				current_row += 1
	
	print(f"{current_row + 1} {current_col + 1}")

if __name__ == "__main__":
	main()