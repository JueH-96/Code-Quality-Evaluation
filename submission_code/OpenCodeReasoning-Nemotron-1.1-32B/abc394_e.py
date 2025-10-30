from collections import deque

def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n = int(data[0].strip())
	grid = []
	for i in range(1, 1+n):
		grid.append(data[i].strip())
	
	in_edges = [[] for _ in range(n)]
	out_edges = [[] for _ in range(n)]
	
	for i in range(n):
		for j in range(n):
			if grid[i][j] != '-':
				out_edges[i].append((j, grid[i][j]))
				in_edges[j].append((i, grid[i][j]))
	
	INF = 10**9
	dp = [[INF] * n for _ in range(n)]
	q = deque()
	
	for i in range(n):
		dp[i][i] = 0
		q.append((i, i))
		
	for i in range(n):
		for j in range(n):
			if grid[i][j] != '-':
				if dp[i][j] > 1:
					dp[i][j] = 1
					q.append((i, j))
					
	while q:
		i, j = q.popleft()
		for k, c1 in in_edges[i]:
			for l, c2 in out_edges[j]:
				if c1 == c2:
					if dp[k][l] > dp[i][j] + 2:
						dp[k][l] = dp[i][j] + 2
						q.append((k, l))
						
	for i in range(n):
		line = []
		for j in range(n):
			if dp[i][j] == INF:
				line.append("-1")
			else:
				line.append(str(dp[i][j]))
		print(" ".join(line))

if __name__ == "__main__":
	main()