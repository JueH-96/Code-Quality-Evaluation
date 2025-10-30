import sys

def main():
	data = sys.stdin.read().strip().split()
	if not data:
		return
	n = int(data[0])
	cuboids = []
	index = 1
	for i in range(n):
		coords = list(map(int, data[index:index+6]))
		index += 6
		cuboids.append(coords)
	
	grid = [[[-1 for _ in range(100)] for _ in range(100)] for _ in range(100)]
	
	for idx, (x1, y1, z1, x2, y2, z2) in enumerate(cuboids):
		for x in range(x1, x2):
			for y in range(y1, y2):
				for z in range(z1, z2):
					if grid[x][y][z] == -1:
						grid[x][y][z] = idx
	
	adj_count = [0] * n
	seen = set()
	directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
	
	for x in range(100):
		for y in range(100):
			for z in range(100):
				if grid[x][y][z] == -1:
					continue
				i = grid[x][y][z]
				for dx, dy, dz in directions:
					nx, ny, nz = x + dx, y + dy, z + dz
					if 0 <= nx < 100 and 0 <= ny < 100 and 0 <= nz < 100:
						j = grid[nx][ny][nz]
						if j == -1 or j == i:
							continue
						a, b = min(i, j), max(i, j)
						if (a, b) not in seen:
							seen.add((a, b))
							adj_count[i] += 1
							adj_count[j] += 1
	
	for count in adj_count:
		print(count)

if __name__ == "__main__":
	main()