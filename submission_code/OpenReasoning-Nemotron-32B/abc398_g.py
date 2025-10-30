import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		print("Takahashi")
		return
	
	n = int(data[0])
	m = int(data[1])
	graph = [[] for _ in range(n+1)]
	index = 2
	for _ in range(m):
		u = int(data[index])
		v = int(data[index+1])
		index += 2
		graph[u].append(v)
		graph[v].append(u)
	
	visited = [False] * (n+1)
	total_xor = 0
	
	for i in range(1, n+1):
		if not visited[i]:
			color = [-1] * (n+1)
			q = deque([i])
			color[i] = 0
			count = [0, 0]
			count[0] = 1
			visited[i] = True
			while q:
				u = q.popleft()
				for v in graph[u]:
					if not visited[v]:
						visited[v] = True
						color[v] = color[u] ^ 1
						count[color[v]] += 1
						q.append(v)
			total_xor ^= count[0] * count[1]
	
	print("Aoki" if total_xor != 0 else "Takahashi")

if __name__ == '__main__':
	main()