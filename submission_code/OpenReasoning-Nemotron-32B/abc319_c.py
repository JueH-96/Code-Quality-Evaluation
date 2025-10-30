import itertools
import math

def main():
	grid = []
	for _ in range(3):
		row = list(map(int, input().split()))
		grid.append(row)
	
	flat = [grid[i][j] for i in range(3) for j in range(3)]
	
	lines = [
		[0, 1, 2], [3, 4, 5], [6, 7, 8],
		[0, 3, 6], [1, 4, 7], [2, 5, 8],
		[0, 4, 8], [2, 4, 6]
	]
	
	dangerous_lines = []
	for line in lines:
		a, b, c = line
		num_a = flat[a]
		num_b = flat[b]
		num_c = flat[c]
		if num_a == num_b == num_c:
			continue
		if len({num_a, num_b, num_c}) == 2:
			if num_a == num_b:
				diff_index = c
			elif num_a == num_c:
				diff_index = b
			else:
				diff_index = a
			dangerous_lines.append((line, diff_index))
	
	total_perm = math.factorial(9)
	good_perm = 0
	
	for perm in itertools.permutations(range(9)):
		pos = [0] * 9
		for order, idx in enumerate(perm):
			pos[idx] = order
		
		valid = True
		for line, diff_index in dangerous_lines:
			p0 = pos[line[0]]
			p1 = pos[line[1]]
			p2 = pos[line[2]]
			max_pos = max(p0, p1, p2)
			if pos[diff_index] == max_pos:
				valid = False
				break
		
		if valid:
			good_perm += 1
	
	probability = good_perm / total_perm
	print("{:.15f}".format(probability))

if __name__ == '__main__':
	main()