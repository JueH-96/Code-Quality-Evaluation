def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	H, W = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(list(data[i].strip()))
	
	removed = [[False] * W for _ in range(H)]
	row_total = [W] * H
	col_total = [H] * W
	
	row_color_count = [dict() for _ in range(H)]
	for i in range(H):
		for j in range(W):
			c = grid[i][j]
			row_color_count[i][c] = row_color_count[i].get(c, 0) + 1
			
	col_color_count = [dict() for _ in range(W)]
	for j in range(W):
		for i in range(H):
			c = grid[i][j]
			col_color_count[j][c] = col_color_count[j].get(c, 0) + 1
			
	while True:
		marked_rows = set()
		marked_cols = set()
		
		for i in range(H):
			if row_total[i] >= 2 and len(row_color_count[i]) == 1:
				marked_rows.add(i)
				
		for j in range(W):
			if col_total[j] >= 2 and len(col_color_count[j]) == 1:
				marked_cols.add(j)
				
		if not marked_rows and not marked_cols:
			break
			
		for i in marked_rows:
			if row_total[i] == 0:
				continue
			for j in range(W):
				if not removed[i][j]:
					removed[i][j] = True
					c = grid[i][j]
					row_color_count[i][c] -= 1
					if row_color_count[i][c] == 0:
						del row_color_count[i][c]
					row_total[i] -= 1
					
					col_color_count[j][c] -= 1
					if col_color_count[j][c] == 0:
						del col_color_count[j][c]
					col_total[j] -= 1
					
		for j in marked_cols:
			if col_total[j] == 0:
				continue
			for i in range(H):
				if not removed[i][j]:
					removed[i][j] = True
					c = grid[i][j]
					row_color_count[i][c] -= 1
					if row_color_count[i][c] == 0:
						del row_color_count[i][c]
					row_total[i] -= 1
					
					col_color_count[j][c] -= 1
					if col_color_count[j][c] == 0:
						del col_color_count[j][c]
					col_total[j] -= 1
					
	count = 0
	for i in range(H):
		for j in range(W):
			if not removed[i][j]:
				count += 1
	print(count)

if __name__ == "__main__":
	main()