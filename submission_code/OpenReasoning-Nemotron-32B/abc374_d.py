import math
import sys
from itertools import permutations, product

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0.0)
		return

	n = int(data[0])
	S_val = int(data[1])
	T_val = int(data[2])
	segments = []
	idx = 3
	for i in range(n):
		a = int(data[idx])
		b = int(data[idx+1])
		c = int(data[idx+2])
		d = int(data[idx+3])
		idx += 4
		segments.append((a, b, c, d))
	
	total_fixed = 0.0
	for seg in segments:
		a, b, c, d = seg
		dx = a - c
		dy = b - d
		length = math.sqrt(dx*dx + dy*dy)
		total_fixed += length / T_val

	if n == 0:
		print(0.0)
		return

	best_total = float('inf')

	for perm in permutations(range(n)):
		for directions in product([0, 1], repeat=n):
			current_x, current_y = 0.0, 0.0
			moving_time = 0.0
			for i in range(n):
				seg_index = perm[i]
				a, b, c, d = segments[seg_index]
				if directions[i] == 0:
					start_x, start_y = a, b
					end_x, end_y = c, d
				else:
					start_x, start_y = c, d
					end_x, end_y = a, b
				
				dx = current_x - start_x
				dy = current_y - start_y
				dist = math.sqrt(dx*dx + dy*dy)
				moving_time += dist / S_val
				
				current_x, current_y = end_x, end_y
				
			total_time = total_fixed + moving_time
			if total_time < best_total:
				best_total = total_time
				
	print(best_total)

if __name__ == "__main__":
	main()