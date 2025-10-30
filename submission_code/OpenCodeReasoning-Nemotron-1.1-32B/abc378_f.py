import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	graph = [[] for _ in range(n+1)]
	deg = [0] * (n+1)
	index = 1
	for _ in range(n-1):
		u = int(data[index])
		v = int(data[index+1])
		index += 2
		graph[u].append(v)
		graph[v].append(u)
		deg[u] += 1
		deg[v] += 1

	comp_id = [0] * (n+1)
	visited = [False] * (n+1)
	comp_index = 0
	for i in range(1, n+1):
		if deg[i] == 3 and not visited[i]:
			comp_index += 1
			q = deque([i])
			visited[i] = True
			comp_id[i] = comp_index
			while q:
				node = q.popleft()
				for neighbor in graph[node]:
					if deg[neighbor] == 3 and not visited[neighbor]:
						visited[neighbor] = True
						comp_id[neighbor] = comp_index
						q.append(neighbor)
	
	comp_set = [set() for _ in range(n+1)]
	for u in range(1, n+1):
		if deg[u] == 2:
			for v in graph[u]:
				if deg[v] == 3:
					cid = comp_id[v]
					if cid != 0:
						comp_set[u].add(cid)
	
	L = [[] for _ in range(comp_index+1)]
	for u in range(1, n+1):
		if deg[u] == 2:
			for cid in comp_set[u]:
				L[cid].append(u)
				
	subtract_count = [0] * (comp_index+1)
	for u in range(1, n+1):
		if deg[u] == 2:
			for v in graph[u]:
				if v > u and deg[v] == 2:
					common = comp_set[u] & comp_set[v]
					for cid in common:
						subtract_count[cid] += 1
						
	total_ans = 0
	for cid in range(1, comp_index+1):
		k = len(L[cid])
		total_pairs = k * (k - 1) // 2
		total_ans += total_pairs - subtract_count[cid]
		
	print(total_ans)

if __name__ == "__main__":
	main()