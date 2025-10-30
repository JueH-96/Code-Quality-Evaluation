import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	n = int(next(it)); m = int(next(it))
	graph = [[] for _ in range(n+1)]
	edges = []
	for i in range(m):
		x = int(next(it)); y = int(next(it)); z = int(next(it))
		graph[x].append((y, z))
		graph[y].append((x, z))
		edges.append((x, y, z))
	
	comp_id = [0] * (n+1)
	comp_list = []
	comp_index = 0
	for i in range(1, n+1):
		if comp_id[i] == 0:
			comp_index += 1
			comp = []
			queue = deque([i])
			comp_id[i] = comp_index
			comp.append(i)
			while queue:
				u = queue.popleft()
				for (v, w) in graph[u]:
					if comp_id[v] == 0:
						comp_id[v] = comp_index
						comp.append(v)
						queue.append(v)
			comp_list.append(comp)
	
	comp_edges_list = [[] for _ in range(comp_index+1)]
	for (x, y, z) in edges:
		cid = comp_id[x]
		comp_edges_list[cid].append((x, y, z))
	
	res = [0] * (n+1)
	
	for comp in comp_list:
		comp_set = set(comp)
		cid = comp_id[comp[0]]
		for j in range(0, 32):
			assignment = {}
			visited = set()
			queue = deque()
			root = comp[0]
			assignment[root] = 0
			visited.add(root)
			queue.append(root)
			while queue:
				u = queue.popleft()
				for (v, w) in graph[u]:
					if v not in comp_set:
						continue
					if v not in visited:
						visited.add(v)
						label = (w >> j) & 1
						assignment[v] = assignment[u] ^ label
						queue.append(v)
			
			for (x, y, z) in comp_edges_list[cid]:
				label = (z >> j) & 1
				if assignment[x] ^ assignment[y] != label:
					print(-1)
					return
					
			count0 = 0
			for node in comp:
				count0 += assignment[node]
			count1 = len(comp) - count0
			if count0 <= count1:
				for node in comp:
					res[node] += assignment[node] * (1 << j)
			else:
				for node in comp:
					res[node] += (1 - assignment[node]) * (1 << j)
					
	print(" ".join(str(res[i]) for i in range(1, n+1)))

if __name__ == '__main__':
	main()