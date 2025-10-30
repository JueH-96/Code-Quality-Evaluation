import bisect
from collections import defaultdict
import sys

def main():
	data = sys.stdin.read().splitlines()
	W, H = map(int, data[0].split())
	N = int(data[1])
	strawberries = []
	for i in range(2, 2 + N):
		p, q = map(int, data[i].split())
		strawberries.append((p, q))
	
	A = int(data[2 + N])
	a_list = list(map(int, data[2 + N + 1].split()))
	B = int(data[2 + N + 2])
	b_list = list(map(int, data[2 + N + 3].split()))
	
	vertical_bounds = [0] + a_list + [W]
	horizontal_bounds = [0] + b_list + [H]
	
	count_dict = defaultdict(int)
	
	for p, q in strawberries:
		i = bisect.bisect_right(vertical_bounds, p) - 1
		j = bisect.bisect_right(horizontal_bounds, q) - 1
		count_dict[(i, j)] += 1
		
	total_pieces = (A + 1) * (B + 1)
	non_empty = len(count_dict)
	
	if non_empty < total_pieces:
		min_val = 0
	else:
		min_val = min(count_dict.values())
		
	max_val = max(count_dict.values())
	
	print(min_val, max_val)

if __name__ == '__main__':
	main()