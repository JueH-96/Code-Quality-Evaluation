def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	
	n, m = map(int, data[0].split())
	grid = []
	for i in range(1, 1 + n):
		grid.append(data[i].strip())
	
	results = []
	for i in range(n - 8):
		for j in range(m - 8):
			valid = True
			
			for r in range(i, i + 3):
				for c in range(j, j + 3):
					if grid[r][c] != '#':
						valid = False
						break
				if not valid:
					break
			
			if not valid:
				continue
				
			for r in range(i + 6, i + 9):
				for c in range(j + 6, j + 9):
					if grid[r][c] != '#':
						valid = False
						break
				if not valid:
					break
					
			if not valid:
				continue
				
			for r in range(i, i + 4):
				for c in range(j + 3, j + 6):
					if grid[r][c] != '.':
						valid = False
						break
				if not valid:
					break
					
			if not valid:
				continue
				
			for r in range(i + 3, i + 6):
				for c in range(j, j + 3):
					if grid[r][c] != '.':
						valid = False
						break
				if not valid:
					break
					
			if not valid:
				continue
				
			for r in range(i + 5, i + 9):
				for c in range(j + 5, j + 9):
					if grid[r][c] != '.':
						valid = False
						break
				if not valid:
					break
					
			if not valid:
				continue
				
			for r in range(i + 6, i + 9):
				for c in range(j + 3, j + 6):
					if grid[r][c] != '.':
						valid = False
						break
				if not valid:
					break
					
			if not valid:
				continue
				
			for r in range(i + 3, i + 6):
				for c in range(j + 6, j + 9):
					if grid[r][c] != '.':
						valid = False
						break
				if not valid:
					break
					
			if valid:
				results.append((i + 1, j + 1))
				
	for res in results:
		print(f"{res[0]} {res[1]}")

if __name__ == "__main__":
	main()