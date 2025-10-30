import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
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
		
	req_list = list(map(int, data[index:index+k]))
	req = [False] * (n+1)
	for v in req_list:
		req[v] = True
		
	alive = [True] * (n+1)
	q = deque()
	for i in range(1, n+1):
		if deg[i] == 1 and not req[i]:
			q.append(i)
			
	remaining = n
	while q:
		u = q.popleft()
		if not alive[u]:
			continue
		alive[u] = False
		remaining -= 1
		for v in graph[u]:
			if alive[v]:
				deg[v] -= 1
				if deg[v] == 1 and not req[v]:
					q.append(v)
					
	print(remaining)

if __name__ == "__main__":
	main()