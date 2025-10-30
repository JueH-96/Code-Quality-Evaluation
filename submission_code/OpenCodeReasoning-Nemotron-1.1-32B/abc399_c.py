import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	m = int(data[1])
	graph = [[] for _ in range(n+1)]
	index = 2
	for i in range(m):
		u = int(data[index])
		v = int(data[index+1])
		index += 2
		graph[u].append(v)
		graph[v].append(u)
	
	visited = [False] * (n+1)
	comp_count = 0
	for i in range(1, n+1):
		if not visited[i]:
			comp_count += 1
			q = deque([i])
			visited[i] = True
			while q:
				node = q.popleft()
				for neighbor in graph[node]:
					if not visited[neighbor]:
						visited[neighbor] = True
						q.append(neighbor)
						
	ans = m - (n - comp_count)
	print(ans)

if __name__ == '__main__':
	main()