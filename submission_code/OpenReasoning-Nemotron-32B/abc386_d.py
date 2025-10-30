import sys
from collections import defaultdict

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print("Yes")
		return
	
	n, m = map(int, data[0].split())
	black = []
	white = []
	rows_black = defaultdict(list)
	cols_black = defaultdict(list)
	rows_white = defaultdict(list)
	cols_white = defaultdict(list)
	
	for i in range(1, m+1):
		parts = data[i].split()
		x = int(parts[0])
		y = int(parts[1])
		c = parts[2]
		if c == 'B':
			black.append((x, y))
			rows_black[x].append(y)
			cols_black[y].append(x)
		else:
			white.append((x, y))
			rows_white[x].append(y)
			cols_white[y].append(x)
	
	rows_with_black = set(rows_black.keys())
	rows_with_white = set(rows_white.keys())
	for x in rows_with_black & rows_with_white:
		max_black = max(rows_black[x])
		min_white = min(rows_white[x])
		if max_black >= min_white:
			print("No")
			return
			
	cols_with_black = set(cols_black.keys())
	cols_with_white = set(cols_white.keys())
	for y in cols_with_black & cols_with_white:
		max_black = max(cols_black[y])
		min_white = min(cols_white[y])
		if max_black >= min_white:
			print("No")
			return
			
	R = rows_with_black | rows_with_white
	C = cols_with_black | cols_with_white
	
	F_dict = {}
	if R:
		if black:
			R_sorted = sorted(R, reverse=True)
			black_sorted = sorted(black, key=lambda cell: cell[0], reverse=True)
			j = 0
			cur_max = 0
			for x in R_sorted:
				while j < len(black_sorted) and black_sorted[j][0] >= x:
					if black_sorted[j][1] > cur_max:
						cur_max = black_sorted[j][1]
					j += 1
				F_dict[x] = cur_max
		else:
			for x in R:
				F_dict[x] = 0
	else:
		F_dict = {}
	
	G_dict = {}
	if C:
		if black:
			C_sorted = sorted(C, reverse=True)
			black_sorted_col = sorted(black, key=lambda cell: cell[1], reverse=True)
			j_col = 0
			cur_max_x = 0
			for y in C_sorted:
				while j_col < len(black_sorted_col) and black_sorted_col[j_col][1] >= y:
					if black_sorted_col[j_col][0] > cur_max_x:
						cur_max_x = black_sorted_col[j_col][0]
					j_col += 1
				G_dict[y] = cur_max_x
		else:
			for y in C:
				G_dict[y] = 0
	else:
		G_dict = {}
	
	for x in rows_white:
		min_white = min(rows_white[x])
		if F_dict[x] >= min_white:
			print("No")
			return
			
	for y in cols_white:
		min_white = min(cols_white[y])
		if G_dict[y] >= min_white:
			print("No")
			return
			
	print("Yes")

if __name__ == "__main__":
	main()