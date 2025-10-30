import sys

mod = 998244353

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	H = int(next(it))
	W = int(next(it))
	grid = []
	for _ in range(H):
		row = list(map(int, [next(it) for _ in range(W)]))
		grid.append(row)
	
	Q = int(next(it))
	sh = int(next(it))
	sw = int(next(it))
	r, c = sh - 1, sw - 1
	
	dp = [[0] * W for _ in range(H)]
	dp[0][0] = grid[0][0] % mod
	
	for i in range(H):
		for j in range(W):
			if i == 0 and j == 0:
				continue
			total = 0
			if i > 0:
				total = (total + dp[i-1][j]) % mod
			if j > 0:
				total = (total + dp[i][j-1]) % mod
			dp[i][j] = total * grid[i][j] % mod

	out_lines = []
	for _ in range(Q):
		d = next(it)
		a_val = int(next(it))
		if d == 'U':
			r -= 1
		elif d == 'D':
			r += 1
		elif d == 'L':
			c -= 1
		elif d == 'R':
			c += 1
		
		grid[r][c] = a_val
		
		for i in range(r, H):
			start_j = c
			for j in range(start_j, W):
				if i == 0 and j == 0:
					dp[0][0] = grid[0][0] % mod
				else:
					total = 0
					if i > 0:
						total = (total + dp[i-1][j]) % mod
					if j > 0:
						total = (total + dp[i][j-1]) % mod
					dp[i][j] = total * grid[i][j] % mod
		
		out_lines.append(str(dp[H-1][W-1] % mod))
	
	print("
".join(out_lines))

if __name__ == '__main__':
	main()