import sys
from collections import deque

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n = int(data[0])
	graph = []
	for i in range(1, 1 + n):
		s = data[i].strip()
		graph.append(list(s))
	
	INF = 10**9
	dp = [[INF] * n for _ in range(n)]
	q = deque()
	
	for i in range(n):
		dp[i][i] = 0
		q.append((i, i))
	
	for i in range(n):
		for j in range(n):
			if graph[i][j] != '-':
				if dp[i][j] > 1:
					dp[i][j] = 1
					q.append((i, j))
	
	while q:
		u, v = q.popleft()
		for x in range(n):
			if graph[x][u] != '-':
				a = graph[x][u]
				for y in range(n):
					if graph[v][y] != '-' and graph[v][y] == a:
						if dp[x][y] > dp[u][v] + 2:
							dp[x][y] = dp[u][v] + 2
							q.append((x, y))
	
	for i in range(n):
		for j in range(n):
			if dp[i][j] == INF:
				dp[i][j] = -1
	
	for i in range(n):
		print(" ".join(str(dp[i][j]) for j in range(n)))

if __name__ == '__main__':
	main()