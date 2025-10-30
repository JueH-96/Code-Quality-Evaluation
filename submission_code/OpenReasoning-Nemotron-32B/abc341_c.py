import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	H, W, N = map(int, data[0].split())
	T = data[1].strip()
	grid = []
	for i in range(2, 2+H):
		grid.append(data[i].strip())
	
	dp = [[False] * W for _ in range(H)]
	for i in range(H):
		for j in range(W):
			if grid[i][j] == '.':
				dp[i][j] = True
				
	for move in T:
		dp_next = [[False] * W for _ in range(H)]
		if move == 'L':
			for i in range(H):
				for j in range(0, W-1):
					if dp[i][j+1] and grid[i][j] == '.':
						dp_next[i][j] = True
		elif move == 'R':
			for i in range(H):
				for j in range(1, W):
					if dp[i][j-1] and grid[i][j] == '.':
						dp_next[i][j] = True
		elif move == 'U':
			for i in range(0, H-1):
				for j in range(W):
					if dp[i+1][j] and grid[i][j] == '.':
						dp_next[i][j] = True
		elif move == 'D':
			for i in range(1, H):
				for j in range(W):
					if dp[i-1][j] and grid[i][j] == '.':
						dp_next[i][j] = True
		dp = dp_next
		
	count = 0
	for i in range(H):
		for j in range(W):
			if dp[i][j]:
				count += 1
	print(count)

if __name__ == '__main__':
	main()