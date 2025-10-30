import bisect

def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		print("No")
		return
	n, m = map(int, data[0].split())
	fixed = []
	row_info = {}
	col_info = {}
	
	for i in range(1, 1 + m):
		parts = data[i].split()
		x = int(parts[0])
		y = int(parts[1])
		c = parts[2]
		fixed.append((x, y, c))
		
		if x not in row_info:
			row_info[x] = [10**18, 0]
		if c == 'W':
			if y < row_info[x][0]:
				row_info[x][0] = y
		else:
			if y > row_info[x][1]:
				row_info[x][1] = y
				
		if y not in col_info:
			col_info[y] = [10**18, 0]
		if c == 'W':
			if x < col_info[y][0]:
				col_info[y][0] = x
		else:
			if x > col_info[y][1]:
				col_info[y][1] = x
				
	rows_sorted = sorted(set([0] + list(row_info.keys()) + [n + 1]))
	F_dict = {}
	F_dict[0] = n
	for i in range(1, len(rows_sorted)):
		r = rows_sorted[i]
		prev_r = rows_sorted[i - 1]
		candidate = F_dict[prev_r]
		if r in row_info:
			min_white, max_black = row_info[r]
			if min_white != 10**18:
				candidate = min(candidate, min_white - 1)
			if candidate < max_black:
				print("No")
				return
		F_dict[r] = candidate
		
	cols_sorted = sorted(set([0] + list(col_info.keys()) + [n + 1]))
	G_dict = {}
	G_dict[0] = n
	for i in range(1, len(cols_sorted)):
		c = cols_sorted[i]
		prev_c = cols_sorted[i - 1]
		candidate = G_dict[prev_c]
		if c in col_info:
			min_white, max_black = col_info[c]
			if min_white != 10**18:
				candidate = min(candidate, min_white - 1)
			if candidate < max_black:
				print("No")
				return
		G_dict[c] = candidate
		
	for (x, y, c) in fixed:
		idx_r = bisect.bisect_right(rows_sorted, x) - 1
		F_x = F_dict[rows_sorted[idx_r]]
		
		idx_c = bisect.bisect_right(cols_sorted, y) - 1
		G_y = G_dict[cols_sorted[idx_c]]
		
		if c == 'B':
			if not (y <= F_x and x <= G_y):
				print("No")
				return
		else:
			if not (y > F_x or x > G_y):
				print("No")
				return
				
	print("Yes")

if __name__ == "__main__":
	main()