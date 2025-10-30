import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	H, W = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(list(data[i].strip()))
	
	removed = [[False] * W for _ in range(H)]
	row_count = [W] * H
	col_count = [H] * W
	row_color_count = [dict() for _ in range(H)]
	col_color_count = [dict() for _ in range(W)]
	
	for i in range(H):
		for j in range(W):
			c = grid[i][j]
			if c in row_color_count[i]:
				row_color_count[i][c] += 1
			else:
				row_color_count[i][c] = 1
			if c in col_color_count[j]:
				col_color_count[j][c] += 1
			else:
				col_color_count[j][c] = 1
				
	dirty_rows = set(range(H))
	dirty_cols = set(range(W))
	
	while dirty_rows or dirty_cols:
		R = set()
		for i in dirty_rows:
			if row_count[i] >= 2 and len(row_color_count[i]) == 1:
				R.add(i)
		dirty_rows = set()
		
		C = set()
		for j in dirty_cols:
			if col_count[j] >= 2 and len(col_color_count[j]) == 1:
				C.add(j)
		dirty_cols = set()
		
		if not R and not C:
			break
			
		for i in R:
			for j in range(W):
				if not removed[i][j]:
					removed[i][j] = True
					c = grid[i][j]
					row_color_count[i][c] -= 1
					if row_color_count[i][c] == 0:
						del row_color_count[i][c]
					row_count[i] -= 1
					
					col_color_count[j][c] -= 1
					if col_color_count[j][c] == 0:
						del col_color_count[j][c]
					col_count[j] -= 1
					dirty_cols.add(j)
					
		for j in C:
			for i in range(H):
				if not removed[i][j]:
					removed[i][j] = True
					c = grid[i][j]
					row_color_count[i][c] -= 1
					if row_color_count[i][c] == 0:
						del row_color_count[i][c]
					row_count[i] -= 1
					dirty_rows.add(i)
					
					col_color_count[j][c] -= 1
					if col_color_count[j][c] == 0:
						del col_color_count[j][c]
					col_count[j] -= 1
					
	ans = sum(row_count)
	print(ans)

if __name__ == "__main__":
	main()