def main():
	import sys
	data = sys.stdin.read().splitlines()
	H, W = map(int, data[0].split())
	grid = []
	for i in range(1, 1 + H):
		grid.append(data[i].strip())
	
	directions = [
		(0, 1),   # right
		(1, 0),	# down
		(0, -1),   # left
		(-1, 0),   # up
		(1, 1),	# down-right
		(1, -1),   # down-left
		(-1, 1),   # up-right
		(-1, -1)   # up-left
	]
	
	for r in range(H):
		for c in range(W):
			if grid[r][c] != 's':
				continue
			for dr, dc in directions:
				positions = []
				current_r, current_c = r, c
				valid = True
				for i, char in enumerate("snuke"):
					if current_r < 0 or current_r >= H or current_c < 0 or current_c >= W:
						valid = False
						break
					if grid[current_r][current_c] != char:
						valid = False
						break
					positions.append((current_r, current_c))
					if i < 4:
						current_r += dr
						current_c += dc
				if valid:
					for (x, y) in positions:
						print(f"{x + 1} {y + 1}")
					return

if __name__ == "__main__":
	main()