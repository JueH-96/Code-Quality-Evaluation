import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	H = int(data[0])
	W = int(data[1])
	N = int(data[2])
	grid = [[False] * W for _ in range(H)]
	
	index = 3
	for _ in range(N):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		grid[a-1][b-1] = True
		
	dp_prev = [0] * W
	total = 0
	
	for i in range(H-1, -1, -1):
		dp_curr = [0] * W
		for j in range(W-1, -1, -1):
			if grid[i][j]:
				dp_curr[j] = 0
			else:
				if i == H-1 or j == W-1:
					dp_curr[j] = 1
				else:
					dp_curr[j] = min(dp_prev[j], dp_curr[j+1], dp_prev[j+1]) + 1
				total += dp_curr[j]
		dp_prev = dp_curr
		
	print(total)

if __name__ == '__main__':
	main()