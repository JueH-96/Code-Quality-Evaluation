import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	first_line = data[0].split()
	N = int(first_line[0])
	Q = int(first_line[1])
	grid = []
	for i in range(1, 1 + N):
		grid.append(data[i].strip())
	
	base = [[0] * N for _ in range(N)]
	for i in range(N):
		for j in range(N):
			if grid[i][j] == 'B':
				base[i][j] = 1
	
	prefix = [[0] * N for _ in range(N)]
	for i in range(N):
		for j in range(N):
			top = prefix[i-1][j] if i-1 >= 0 else 0
			left = prefix[i][j-1] if j-1 >= 0 else 0
			diag = prefix[i-1][j-1] if (i-1 >= 0 and j-1 >= 0) else 0
			prefix[i][j] = base[i][j] + top + left - diag
	
	base_total = prefix[N-1][N-1] if N > 0 else 0

	def get_rect_base(i1, i2, j1, j2):
		if i1 > i2 or j1 > j2:
			return 0
		total = prefix[i2][j2]
		if i1 > 0:
			total -= prefix[i1-1][j2]
		if j1 > 0:
			total -= prefix[i2][j1-1]
		if i1 > 0 and j1 > 0:
			total += prefix[i1-1][j1-1]
		return total

	def F(x, y):
		if x < 0 or y < 0:
			return 0
		rows = x + 1
		cols = y + 1
		full_rows = rows // N
		full_cols = cols // N
		rem_rows = rows % N
		rem_cols = cols % N
		
		total = full_rows * full_cols * base_total
		
		if rem_rows > 0:
			total += full_cols * get_rect_base(0, rem_rows-1, 0, N-1)
		if rem_cols > 0:
			total += full_rows * get_rect_base(0, N-1, 0, rem_cols-1)
		if rem_rows > 0 and rem_cols > 0:
			total += get_rect_base(0, rem_rows-1, 0, rem_cols-1)
		
		return total

	out_lines = []
	for i in range(1 + N, 1 + N + Q):
		parts = data[i].split()
		A = int(parts[0])
		B = int(parts[1])
		C = int(parts[2])
		D = int(parts[3])
		ans = F(C, D) - F(A-1, D) - F(C, B-1) + F(A-1, B-1)
		out_lines.append(str(ans))
	
	print("
".join(out_lines))

if __name__ == '__main__':
	main()