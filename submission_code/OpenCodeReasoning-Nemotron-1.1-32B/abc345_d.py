import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print("No")
		return
	
	n = int(data[0])
	H = int(data[1])
	W = int(data[2])
	tiles = []
	idx = 3
	for i in range(n):
		a = int(data[idx])
		b = int(data[idx+1])
		idx += 2
		tiles.append((a, b))
	
	total_area = H * W

	def dfs(r, c, tile_list, grid):
		if r == H:
			return True
		if c == W:
			return dfs(r+1, 0, tile_list, grid)
		if grid[r][c]:
			return dfs(r, c+1, tile_list, grid)
			
		if len(tile_list) == 0:
			return False
			
		for idx_val in range(len(tile_list)):
			a, b = tile_list[idx_val]
			if r + a <= H and c + b <= W:
				valid = True
				for i1 in range(r, r+a):
					for j1 in range(c, c+b):
						if grid[i1][j1]:
							valid = False
							break
					if not valid:
						break
				if valid:
					for i1 in range(r, r+a):
						for j1 in range(c, c+b):
							grid[i1][j1] = True
					new_list = tile_list[:idx_val] + tile_list[idx_val+1:]
					if dfs(r, c+1, new_list, grid):
						return True
					for i1 in range(r, r+a):
						for j1 in range(c, c+b):
							grid[i1][j1] = False
							
			if r + b <= H and c + a <= W:
				valid = True
				for i1 in range(r, r+b):
					for j1 in range(c, c+a):
						if grid[i1][j1]:
							valid = False
							break
					if not valid:
						break
				if valid:
					for i1 in range(r, r+b):
						for j1 in range(c, c+a):
							grid[i1][j1] = True
					new_list = tile_list[:idx_val] + tile_list[idx_val+1:]
					if dfs(r, c+1, new_list, grid):
						return True
					for i1 in range(r, r+b):
						for j1 in range(c, c+a):
							grid[i1][j1] = False
							
		return False

	found = False
	for bitmask in range(1 << n):
		area = 0
		tile_list_sub = []
		for i in range(n):
			if bitmask & (1 << i):
				a, b = tiles[i]
				area += a * b
				tile_list_sub.append((a, b))
				
		if area != total_area:
			continue
			
		grid = [[False] * W for _ in range(H)]
		if dfs(0, 0, tile_list_sub, grid):
			found = True
			break
			
	print("Yes" if found else "No")

if __name__ == "__main__":
	main()