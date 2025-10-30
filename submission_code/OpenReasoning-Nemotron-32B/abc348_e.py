import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	graph = [[] for _ in range(n)]
	index = 1
	for _ in range(n-1):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		a0 = a - 1
		b0 = b - 1
		graph[a0].append(b0)
		graph[b0].append(a0)
	
	C = list(map(int, data[index:index+n]))
	total = sum(C)
	
	parent = [-1] * n
	depth = [0] * n
	size = [0] * n
	for i in range(n):
		size[i] = C[i]
	
	q = deque([0])
	parent[0] = -1
	order = []
	while q:
		u = q.popleft()
		order.append(u)
		for v in graph[u]:
			if v == parent[u]:
				continue
			parent[v] = u
			depth[v] = depth[u] + 1
			q.append(v)
	
	for i in range(len(order)-1, -1, -1):
		u = order[i]
		if parent[u] != -1:
			size[parent[u]] += size[u]
	
	base = 0
	for i in range(n):
		base += depth[i] * C[i]
	
	f = [0] * n
	f[0] = base
	q = deque([0])
	while q:
		u = q.popleft()
		for v in graph[u]:
			if v == parent[u]:
				continue
			f[v] = f[u] + total - 2 * size[v]
			q.append(v)
	
	ans = min(f)
	print(ans)

if __name__ == "__main__":
	main()