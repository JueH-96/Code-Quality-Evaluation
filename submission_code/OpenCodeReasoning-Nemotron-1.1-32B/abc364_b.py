def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	
	H, W = map(int, data[0].split())
	start_line = data[1].split()
	s_i = int(start_line[0])
	s_j = int(start_line[1])
	
	grid = []
	for i in range(2, 2 + H):
		grid.append(data[i].strip())
	
	X = data[2 + H].strip()
	
	r, c = s_i - 1, s_j - 1
	
	for move in X:
		if move == 'L':
			if c - 1 >= 0 and grid[r][c - 1] == '.':
				c -= 1
		elif move == 'R':
			if c + 1 < W and grid[r][c + 1] == '.':
				c += 1
		elif move == 'U':
			if r - 1 >= 0 and grid[r - 1][c] == '.':
				r -= 1
		elif move == 'D':
			if r + 1 < H and grid[r + 1][c] == '.':
				r += 1
	
	print(f"{r + 1} {c + 1}")

if __name__ == "__main__":
	main()