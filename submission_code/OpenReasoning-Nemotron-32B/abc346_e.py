import sys
import bisect
from collections import defaultdict

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	H = int(next(it)); W = int(next(it)); M = int(next(it))
	row_time = [0] * (H + 1)
	row_color = [0] * (H + 1)
	col_time = [0] * (W + 1)
	col_color = [0] * (W + 1)
	
	for idx in range(1, M + 1):
		t = int(next(it))
		a = int(next(it))
		x = int(next(it))
		if t == 1:
			row_time[a] = idx
			row_color[a] = x
		else:
			col_time[a] = idx
			col_color[a] = x
			
	all_col_times = col_time[1:]
	all_col_times.sort()
	all_row_times = row_time[1:]
	all_row_times.sort()
	
	row_dict = defaultdict(list)
	col_dict = defaultdict(list)
	
	for i in range(1, H + 1):
		c = row_color[i]
		t_val = row_time[i]
		row_dict[c].append(t_val)
		
	for j in range(1, W + 1):
		c = col_color[j]
		t_val = col_time[j]
		col_dict[c].append(t_val)
		
	colors = set(row_dict.keys()) | set(col_dict.keys())
	results = []
	
	for c in colors:
		total = 0
		for t in row_dict[c]:
			pos = bisect.bisect_left(all_col_times, t)
			total += pos
			
		for t in col_dict[c]:
			pos = bisect.bisect_right(all_row_times, t)
			total += pos
			
		if total > 0:
			results.append((c, total))
			
	results.sort(key=lambda x: x[0])
	print(len(results))
	for c, cnt in results:
		print(f"{c} {cnt}")

if __name__ == '__main__':
	main()