import sys
import heapq
from collections import defaultdict

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	w = int(next(it))
	blocks = []
	for i in range(n):
		x = int(next(it))
		y = int(next(it))
		blocks.append((x, y, i))
	
	cols = defaultdict(list)
	for x, y, i in blocks:
		cols[x].append((y, i))
	
	min_len = 10**9
	for x in range(1, w + 1):
		if x in cols:
			if len(cols[x]) < min_len:
				min_len = len(cols[x])
		else:
			min_len = 0
	R = min_len

	for x in cols:
		cols[x].sort(key=lambda t: t[0])
	
	S = [0] * R if R > 0 else []
	if R > 0:
		heap = []
		for x in range(1, w + 1):
			if x in cols and cols[x]:
				y_val, block_idx = cols[x][0]
				heapq.heappush(heap, (-y_val, x, 0))
		for j in range(R):
			neg_y, x, idx = heapq.heappop(heap)
			S[j] = -neg_y
			if idx + 1 < len(cols[x]) and idx + 1 < R:
				y_val, block_idx = cols[x][idx + 1]
				heapq.heappush(heap, (-y_val, x, idx + 1))
	
	remove_time = [10**18] * n
	for x in cols:
		col_list = cols[x]
		for j in range(len(col_list)):
			y_val, block_idx = col_list[j]
			if j < R:
				remove_time[block_idx] = S[j]
	
	q = int(next(it))
	out_lines = []
	for _ in range(q):
		T = int(next(it))
		A = int(next(it))
		block_idx = A - 1
		x = blocks[block_idx][0]
		y = blocks[block_idx][1]
		if T < y - 1:
			out_lines.append("Yes")
		else:
			if remove_time[block_idx] > T:
				out_lines.append("Yes")
			else:
				out_lines.append("No")
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()