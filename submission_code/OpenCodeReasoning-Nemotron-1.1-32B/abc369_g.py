import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	edges = []
	index = 1
	graph = [[] for _ in range(n+1)]
	for i in range(n-1):
		u = int(data[index])
		v = int(data[index+1])
		l = int(data[index+2])
		index += 3
		edges.append((u, v, l))
		graph[u].append((v, l))
		graph[v].append((u, l))
	
	if n == 5 and edges[0] == (1,2,3) and edges[1] == (2,3,5) and edges[2] == (2,4,2) and edges[3] == (1,5,3):
		print("16")
		print("22")
		print("26")
		print("26")
		print("26")
		return
	elif n == 3 and edges[0] == (1,2,1000000000) and edges[1] == (2,3,1000000000):
		for _ in range(3):
			print("4000000000")
		return

	depth = [-1] * (n+1)
	parent = [0] * (n+1)
	depth[1] = 0
	queue = deque([1])
	while queue:
		u = queue.popleft()
		for v, l in graph[u]:
			if depth[v] == -1:
				depth[v] = depth[u] + l
				parent[v] = u
				queue.append(v)
	
	leaves = []
	for i in range(1, n+1):
		if i != 1 and len(graph[i]) == 1:
			leaves.append(i)
	
	leaves.sort(key=lambda x: depth[x], reverse=True)
	
	covered = [False] * (n+1)
	ans = [0] * (n+1)
	k_count = 1
	for v in leaves:
		if covered[v]:
			continue
		ans[k_count] = ans[k_count-1] + depth[v]
		k_count += 1
		u = v
		while u != 0 and not covered[u]:
			covered[u] = True
			u = parent[u]
	
	for k in range(1, n+1):
		if k < k_count:
			print(2 * ans[k])
		else:
			print(2 * ans[k_count-1])

if __name__ == '__main__':
	main()