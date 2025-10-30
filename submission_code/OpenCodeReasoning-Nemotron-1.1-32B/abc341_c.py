def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data: 
		print(0)
		return
	H, W, N = map(int, data[0].split())
	T = data[1].strip()
	grid = []
	for i in range(2, 2+H):
		grid.append(data[i].strip())
	
	current = []
	for i in range(1, H-1):
		for j in range(1, W-1):
			if grid[i][j] == '.':
				current.append((i, j))
				
	for move in T:
		if not current:
			break
		next_list = []
		for (i, j) in current:
			ni, nj = i, j
			if move == 'L':
				nj -= 1
			elif move == 'R':
				nj += 1
			elif move == 'U':
				ni -= 1
			elif move == 'D':
				ni += 1
			if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
				next_list.append((ni, nj))
		current = next_list
		
	print(len(current))

if __name__ == '__main__':
	main()