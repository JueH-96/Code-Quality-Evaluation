def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		print("No")
		return
	
	H_A, W_A = map(int, data[0].split())
	A_grid = data[1:1+H_A]
	H_B, W_B = map(int, data[1+H_A].split())
	B_grid = data[2+H_A:2+H_A+H_B]
	H_X, W_X = map(int, data[2+H_A+H_B].split())
	X_grid = data[3+H_A+H_B:3+H_A+H_B+H_X]
	
	min_row_A = float('inf')
	max_row_A = -1
	min_col_A = float('inf')
	max_col_A = -1
	for i in range(H_A):
		for j in range(W_A):
			if A_grid[i][j] == '#':
				min_row_A = min(min_row_A, i)
				max_row_A = max(max_row_A, i)
				min_col_A = min(min_col_A, j)
				max_col_A = max(max_col_A, j)
				
	min_row_B = float('inf')
	max_row_B = -1
	min_col_B = float('inf')
	max_col_B = -1
	for i in range(H_B):
		for j in range(W_B):
			if B_grid[i][j] == '#':
				min_row_B = min(min_row_B, i)
				max_row_B = max(max_row_B, i)
				min_col_B = min(min_col_B, j)
				max_col_B = max(max_col_B, j)
				
	for dx in range(-20, 21):
		for dy in range(-20, 21):
			min_row = min(min_row_A, min_row_B + dy)
			max_row = max(max_row_A, max_row_B + dy)
			min_col = min(min_col_A, min_col_B + dx)
			max_col = max(max_col_A, max_col_B + dx)
			
			h = max_row - min_row + 1
			w = max_col - min_col + 1
			if h > H_X or w > W_X:
				continue
				
			for r0 in range(min_row - (H_X - h), min_row + 1):
				for c0 in range(min_col - (W_X - w), min_col + 1):
					grid = [['.' for _ in range(W_X)] for _ in range(H_X)]
					
					for i in range(H_A):
						for j in range(W_A):
							if A_grid[i][j] == '#':
								abs_r = i
								abs_c = j
								r_idx = abs_r - r0
								c_idx = abs_c - c0
								if 0 <= r_idx < H_X and 0 <= c_idx < W_X:
									grid[r_idx][c_idx] = '#'
					
					for i in range(H_B):
						for j in range(W_B):
							if B_grid[i][j] == '#':
								abs_r = dx + i
								abs_c = dy + j
								r_idx = abs_r - r0
								c_idx = abs_c - c0
								if 0 <= r_idx < H_X and 0 <= c_idx < W_X:
									grid[r_idx][c_idx] = '#'
					
					match = True
					for i in range(H_X):
						for j in range(W_X):
							if grid[i][j] != X_grid[i][j]:
								match = False
								break
						if not match:
							break
					if match:
						print("Yes")
						return
						
	print("No")

if __name__ == "__main__":
	main()