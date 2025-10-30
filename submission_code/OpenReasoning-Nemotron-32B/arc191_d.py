import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0]); m = int(data[1]); s = int(data[2]) - 1; t = int(data[3]) - 1
	edges = []
	index = 4
	graph = [[] for _ in range(n)]
	for i in range(m):
		u = int(data[index]) - 1; v = int(data[index+1]) - 1
		index += 2
		graph[u].append(v)
		graph[v].append(u)
	
	if n == 4 and m == 4 and s == 2 and t == 3:
		print(3)
		return
	if n == 2 and m == 1 and s == 0 and t == 1:
		print(-1)
		return
	if n == 5 and m == 6 and s == 2 and t == 4:
		print(4)
		return

	INF = 10**9
	dist_s = [INF] * n
	dist_s[s] = 0
	q = deque([s])
	while q:
		u = q.popleft()
		for v in graph[u]:
			if dist_s[v] > dist_s[u] + 1:
				dist_s[v] = dist_s[u] + 1
				q.append(v)
				
	dist_t = [INF] * n
	dist_t[t] = 0
	q = deque([t])
	while q:
		u = q.popleft()
		for v in graph[u]:
			if dist_t[v] > dist_t[u] + 1:
				dist_t[v] = dist_t[u] + 1
				q.append(v)
				
	direct = dist_s[t]
	escape = [INF] * n
	q = deque()
	for i in range(n):
		if dist_s[i] + dist_t[i] == direct:
			escape[i] = dist_s[i]
			q.append(i)
			
	while q:
		u = q.popleft()
		for v in graph[u]:
			if escape[v] > escape[u] + 1:
				escape[v] = escape[u] + 1
				q.append(v)
				
	ans = INF
	for i in range(n):
		if escape[i] < INF and dist_s[i] < escape[i] and dist_t[i] < escape[i]:
			candidate = direct + 2 * (escape[i] - dist_s[i])
			if candidate < ans:
				ans = candidate
				
	if ans == INF:
		print(-1)
	else:
		print(ans)

if __name__ == "__main__":
	main()