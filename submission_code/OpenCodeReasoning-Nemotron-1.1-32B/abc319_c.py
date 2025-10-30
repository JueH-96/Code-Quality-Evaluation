import itertools
import math

def main():
	grid = []
	for i in range(3):
		data = input().split()
		row = list(map(int, data))
		grid.append(row)
	
	grid_flat = []
	for i in range(3):
		for j in range(3):
			grid_flat.append(grid[i][j])
			
	lines = [
		(0, 1, 2), (3, 4, 5), (6, 7, 8),
		(0, 3, 6), (1, 4, 7), (2, 5, 8),
		(0, 4, 8), (2, 4, 6)
	]
	
	critical_lines = []
	for line in lines:
		nums = [grid_flat[i] for i in line]
		if len(set(nums)) == 2:
			critical_lines.append(line)
			
	total_perm = math.factorial(9)
	good_count = 0
	
	for perm in itertools.permutations(range(9)):
		pos = [0] * 9
		for idx, cell in enumerate(perm):
			pos[cell] = idx
			
		valid = True
		for line in critical_lines:
			a, b, c = line
			cells_sorted = sorted([a, b, c], key=lambda x: pos[x])
			n0 = grid_flat[cells_sorted[0]]
			n1 = grid_flat[cells_sorted[1]]
			n2 = grid_flat[cells_sorted[2]]
			if n0 == n1 and n0 != n2:
				valid = False
				break
				
		if valid:
			good_count += 1
			
	prob = good_count / total_perm
	print("{:.20f}".format(prob))

if __name__ == "__main__":
	main()