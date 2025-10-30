import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	cuboids = []
	index = 1
	for i in range(n):
		coords = list(map(int, data[index:index+6]))
		index += 6
		cuboids.append(coords)
	
	grid = [[[-1] * 101 for _ in range(101)] for __ in range(101)]
	
	for i, (x1, y1, z1, x2, y2, z2) in enumerate(cuboids):
		for x in range(x1, x2+1):
			for y in range(y1, y2+1):
				for z in range(z1, z2+1):
					if grid[x][y][z] == -1:
						grid[x][y][z] = i
	
	ans = [0] * n
	
	for i in range(n):
		x1, y1, z1, x2, y2, z2 = cuboids[i]
		adjacent_set = set()
		
		for y in range(y1, y2+1):
			for z in range(z1, z2+1):
				if x2+1 < 101:
					j = grid[x2+1][y][z]
					if j != -1 and j != i:
						adjacent_set.add(j)
		
		for y in range(y1, y2+1):
			for z in range(z1, z2+1):
				if x1-1 >= 0:
					j = grid[x1-1][y][z]
					if j != -1 and j != i:
						adjacent_set.add(j)
		
		for x in range(x1, x2+1):
			for z in range(z1, z2+1):
				if y2+1 < 101:
					j = grid[x][y2+1][z]
					if j != -1 and j != i:
						adjacent_set.add(j)
		
		for x in range(x1, x2+1):
			for z in range(z1, z2+1):
				if y1-1 >= 0:
					j = grid[x][y1-1][z]
					if j != -1 and j != i:
						adjacent_set.add(j)
		
		for x in range(x1, x2+1):
			for y in range(y1, y2+1):
				if z2+1 < 101:
					j = grid[x][y][z2+1]
					if j != -1 and j != i:
						adjacent_set.add(j)
		
		for x in range(x1, x2+1):
			for y in range(y1, y2+1):
				if z1-1 >= 0:
					j = grid[x][y][z1-1]
					if j != -1 and j != i:
						adjacent_set.add(j)
		
		final_adjacent = set()
		for j in adjacent_set:
			x1_i, y1_i, z1_i, x2_i, y2_i, z2_i = cuboids[i]
			x1_j, y1_j, z1_j, x2_j, y2_j, z2_j = cuboids[j]
			if x2_i == x1_j:
				y_low = max(y1_i, y1_j)
				y_high = min(y2_i, y2_j)
				z_low = max(z1_i, z1_j)
				z_high = min(z2_i, z2_j)
				if y_low < y_high and z_low < z_high:
					final_adjacent.add(j)
					continue
			if x1_i == x2_j:
				y_low = max(y1_i, y1_j)
				y_high = min(y2_i, y2_j)
				z_low = max(z1_i, z1_j)
				z_high = min(z2_i, z2_j)
				if y_low < y_high and z_low < z_high:
					final_adjacent.add(j)
					continue
			if y2_i == y1_j:
				x_low = max(x1_i, x1_j)
				x_high = min(x2_i, x2_j)
				z_low = max(z1_i, z1_j)
				z_high = min(z2_i, z2_j)
				if x_low < x_high and z_low < z_high:
					final_adjacent.add(j)
					continue
			if y1_i == y2_j:
				x_low = max(x1_i, x1_j)
				x_high = min(x2_i, x2_j)
				z_low = max(z1_i, z1_j)
				z_high = min(z2_i, z2_j)
				if x_low < x_high and z_low < z_high:
					final_adjacent.add(j)
					continue
			if z2_i == z1_j:
				x_low = max(x1_i, x1_j)
				x_high = min(x2_i, x2_j)
				y_low = max(y1_i, y1_j)
				y_high = min(y2_i, y2_j)
				if x_low < x_high and y_low < y_high:
					final_adjacent.add(j)
					continue
			if z1_i == z2_j:
				x_low = max(x1_i, x1_j)
				x_high = min(x2_i, x2_j)
				y_low = max(y1_i, y1_j)
				y_high = min(y2_i, y2_j)
				if x_low < x_high and y_low < y_high:
					final_adjacent.add(j)
					continue
		ans[i] = len(final_adjacent)
	
	for a in ans:
		print(a)

if __name__ == "__main__":
	main()