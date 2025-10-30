def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		print("No")
		return
	
	idx = 0
	H_A, W_A = map(int, data[idx].split())
	idx += 1
	A = []
	for i in range(H_A):
		A.append(data[idx].strip())
		idx += 1
		
	H_B, W_B = map(int, data[idx].split())
	idx += 1
	B = []
	for i in range(H_B):
		B.append(data[idx].strip())
		idx += 1
		
	H_X, W_X = map(int, data[idx].split())
	idx += 1
	X = []
	for i in range(H_X):
		X.append(data[idx].strip())
		idx += 1
		
	A_blacks = []
	for i in range(H_A):
		for j in range(W_A):
			if A[i][j] == '#':
				A_blacks.append((i, j))
				
	B_blacks = []
	for i in range(H_B):
		for j in range(W_B):
			if B[i][j] == '#':
				B_blacks.append((i, j))
				
	if not A_blacks or not B_blacks:
		print("No")
		return
		
	a_i_min = max(-i for (i, j) in A_blacks)
	a_i_max = min(H_X - i - 1 for (i, j) in A_blacks)
	a_j_min = max(-j for (i, j) in A_blacks)
	a_j_max = min(W_X - j - 1 for (i, j) in A_blacks)
	
	b_i_min = max(-i for (i, j) in B_blacks)
	b_i_max = min(H_X - i - 1 for (i, j) in B_blacks)
	b_j_min = max(-j for (i, j) in B_blacks)
	b_j_max = min(W_X - j - 1 for (i, j) in B_blacks)
	
	if a_i_min > a_i_max or a_j_min > a_j_max or b_i_min > b_i_max or b_j_min > b_j_max:
		print("No")
		return
		
	found = False
	for a_i in range(a_i_min, a_i_max + 1):
		for a_j in range(a_j_min, a_j_max + 1):
			for b_i in range(b_i_min, b_i_max + 1):
				for b_j in range(b_j_min, b_j_max + 1):
					grid = [['.' for _ in range(W_X)] for _ in range(H_X)]
					for (i, j) in A_blacks:
						r = a_i + i
						c = a_j + j
						if 0 <= r < H_X and 0 <= c < W_X:
							grid[r][c] = '#'
					for (i, j) in B_blacks:
						r = b_i + i
						c = b_j + j
						if 0 <= r < H_X and 0 <= c < W_X:
							grid[r][c] = '#'
					match = True
					for i in range(H_X):
						s = ''.join(grid[i])
						if s != X[i]:
							match = False
							break
					if match:
						print("Yes")
						return
	print("No")

if __name__ == "__main__":
	main()