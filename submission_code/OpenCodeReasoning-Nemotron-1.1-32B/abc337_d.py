import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(-1)
		return
	H, W, K = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(data[i].strip())
	
	A = [[0] * W for _ in range(H)]
	B = [[0] * W for _ in range(H)]
	
	for i in range(H):
		for j in range(W):
			if grid[i][j] == '.':
				A[i][j] = 1
			elif grid[i][j] == 'x':
				B[i][j] = 1
				
	row_prefix_A = [[0] * W for _ in range(H)]
	row_prefix_B = [[0] * W for _ in range(H)]
	for i in range(H):
		for j in range(W):
			if j == 0:
				row_prefix_A[i][j] = A[i][j]
				row_prefix_B[i][j] = B[i][j]
			else:
				row_prefix_A[i][j] = row_prefix_A[i][j-1] + A[i][j]
				row_prefix_B[i][j] = row_prefix_B[i][j-1] + B[i][j]
				
	col_prefix_A = [[0] * H for _ in range(W)]
	col_prefix_B = [[0] * H for _ in range(W)]
	for j in range(W):
		for i in range(H):
			if i == 0:
				col_prefix_A[j][i] = A[i][j]
				col_prefix_B[j][i] = B[i][j]
			else:
				col_prefix_A[j][i] = col_prefix_A[j][i-1] + A[i][j]
				col_prefix_B[j][i] = col_prefix_B[j][i-1] + B[i][j]
				
	min_ops = 10**9
	
	for i in range(H):
		for j in range(0, W - K + 1):
			if j == 0:
				total_x = row_prefix_B[i][j+K-1]
			else:
				total_x = row_prefix_B[i][j+K-1] - row_prefix_B[i][j-1]
			if total_x > 0:
				continue
				
			if j == 0:
				cost = row_prefix_A[i][j+K-1]
			else:
				cost = row_prefix_A[i][j+K-1] - row_prefix_A[i][j-1]
			if cost < min_ops:
				min_ops = cost
				
	for j in range(W):
		for i in range(0, H - K + 1):
			if i == 0:
				total_x = col_prefix_B[j][i+K-1]
			else:
				total_x = col_prefix_B[j][i+K-1] - col_prefix_B[j][i-1]
			if total_x > 0:
				continue
				
			if i == 0:
				cost = col_prefix_A[j][i+K-1]
			else:
				cost = col_prefix_A[j][i+K-1] - col_prefix_A[j][i-1]
			if cost < min_ops:
				min_ops = cost
				
	if min_ops == 10**9:
		print(-1)
	else:
		print(min_ops)

if __name__ == "__main__":
	main()