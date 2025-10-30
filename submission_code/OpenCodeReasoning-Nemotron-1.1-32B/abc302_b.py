def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	H, W = map(int, data[0].split())
	grid = []
	for i in range(1, 1 + H):
		grid.append(data[i].strip())
	
	dirs = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
	
	found = False
	for i in range(H):
		for j in range(W):
			if grid[i][j] == 's':
				for dx, dy in dirs:
					positions = []
					valid = True
					for step in range(5):
						ni = i + step * dx
						nj = j + step * dy
						if ni < 0 or ni >= H or nj < 0 or nj >= W:
							valid = False
							break
						positions.append((ni, nj))
					if not valid:
						continue
					letters = [grid[x][y] for x, y in positions]
					if letters == ['s', 'n', 'u', 'k', 'e']:
						for x, y in positions:
							print(f"{x + 1} {y + 1}")
						found = True
						break
				if found:
					break
		if found:
			break

if __name__ == '__main__':
	main()