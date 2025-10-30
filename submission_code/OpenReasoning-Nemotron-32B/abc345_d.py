import sys

def main():
	data = sys.stdin.read().strip().split()
	if not data:
		print("No")
		return

	n = int(data[0])
	H = int(data[1])
	W = int(data[2])
	tiles = []
	index = 3
	for i in range(n):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		tiles.append((a, b))
		
	total_area = H * W
	found = False

	for bitmask in range(1 << n):
		subset_tiles = []
		area = 0
		for i in range(n):
			if bitmask & (1 << i):
				a, b = tiles[i]
				area += a * b
				subset_tiles.append((a, b))
		if area != total_area:
			continue
			
		subset_tiles.sort(key=lambda x: x[0] * x[1], reverse=True)
		grid = [[0] * W for _ in range(H)]
		
		if dfs(grid, subset_tiles):
			found = True
			break
			
	print("Yes" if found else "No")

def dfs(grid, tiles):
	if not tiles:
		for i in range(len(grid)):
			for j in range(len(grid[0])):
				if grid[i][j] == 0:
					return False
		return True
		
	H = len(grid)
	W = len(grid[0])
	start_i, start_j = -1, -1
	for i in range(H):
		for j in range(W):
			if grid[i][j] == 0:
				start_i, start_j = i, j
				break
		if start_i != -1:
			break
			
	if start_i == -1:
		return False
		
	for idx in range(len(tiles)):
		a, b = tiles[idx]
		if a <= H - start_i and b <= W - start_j:
			valid = True
			for x in range(start_i, start_i + a):
				for y in range(start_j, start_j + b):
					if grid[x][y] != 0:
						valid = False
						break
				if not valid:
					break
					
			if valid:
				for x in range(start_i, start_i + a):
					for y in range(start_j, start_j + b):
						grid[x][y] = 1
				new_tiles = tiles[:idx] + tiles[idx+1:]
				if dfs(grid, new_tiles):
					return True
				for x in range(start_i, start_i + a):
					for y in range(start_j, start_j + b):
						grid[x][y] = 0
						
		if a != b:
			if b <= H - start_i and a <= W - start_j:
				valid = True
				for x in range(start_i, start_i + b):
					for y in range(start_j, start_j + a):
						if grid[x][y] != 0:
							valid = False
							break
					if not valid:
						break
				if valid:
					for x in range(start_i, start_i + b):
						for y in range(start_j, start_j + a):
							grid[x][y] = 1
					new_tiles = tiles[:idx] + tiles[idx+1:]
					if dfs(grid, new_tiles):
						return True
					for x in range(start_i, start_i + b):
						for y in range(start_j, start_j + a):
							grid[x][y] = 0
	return False

if __name__ == "__main__":
	main()