import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	n = int(data[0])
	k = int(data[1])
	graph = [[] for _ in range(n+1)]
	deg = [0] * (n+1)
	index = 2
	for _ in range(n-1):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		graph[a].append(b)
		graph[b].append(a)
		deg[a] += 1
		deg[b] += 1
	
	terminals = list(map(int, data[index:index+k]))
	is_terminal = [False] * (n+1)
	for t in terminals:
		is_terminal[t] = True
	
	q = deque()
	for i in range(1, n+1):
		if deg[i] == 1 and not is_terminal[i]:
			q.append(i)
	
	removed_count = 0
	while q:
		u = q.popleft()
		removed_count += 1
		deg[u] = 0
		for v in graph[u]:
			if deg[v] > 0:
				deg[v] -= 1
				if deg[v] == 1 and not is_terminal[v]:
					q.append(v)
	
	print(n - removed_count)

if __name__ == "__main__":
	main()