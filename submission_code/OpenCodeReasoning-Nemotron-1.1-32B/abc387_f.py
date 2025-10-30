import sys
from collections import deque

mod = 998244353

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	next_arr = [a-1 for a in A]
	
	visited = [False] * n
	in_cycle = [False] * n
	for i in range(n):
		if not visited[i]:
			path = []
			cur = i
			while not visited[cur]:
				visited[cur] = True
				path.append(cur)
				cur = next_arr[cur]
			if cur in path:
				idx = path.index(cur)
				cycle_nodes = path[idx:]
				for node in cycle_nodes:
					in_cycle[node] = True
					
	rev_graph = [[] for _ in range(n)]
	for i in range(n):
		rev_graph[next_arr[i]].append(i)
		
	tree_rev_graph = []
	for u in range(n):
		tree_rev_graph.append([w for w in rev_graph[u] if not in_cycle[w]])
		
	F = [None] * n
	deg = [0] * n
	for u in range(n):
		if not in_cycle[u]:
			deg[u] = len(tree_rev_graph[u])
			
	q = deque()
	for u in range(n):
		if not in_cycle[u] and deg[u] == 0:
			q.append(u)
			
	while q:
		u = q.popleft()
		g = [1] * (m+1)
		for w in tree_rev_graph[u]:
			for y in range(1, m+1):
				g[y] = (g[y] * F[w][y]) % mod
				
		F[u] = [0] * (m+1)
		for x in range(1, m+1):
			F[u][x] = (F[u][x-1] + g[x]) % mod
			
		for v in rev_graph[u]:
			if not in_cycle[v]:
				deg[v] -= 1
				if deg[v] == 0:
					q.append(v)
					
	visited_comp = [False] * n
	ans = 1
	for i in range(n):
		if in_cycle[i] and not visited_comp[i]:
			cycle_nodes = []
			cur = i
			while not visited_comp[cur]:
				visited_comp[cur] = True
				cycle_nodes.append(cur)
				cur = next_arr[cur]
				
			comp_nodes = set(cycle_nodes)
			q_comp = deque(cycle_nodes)
			while q_comp:
				cur = q_comp.popleft()
				for w in rev_graph[cur]:
					if w not in comp_nodes:
						comp_nodes.add(w)
						q_comp.append(w)
						
			total_comp = 0
			for v_val in range(1, m+1):
				product = 1
				for u in comp_nodes:
					if in_cycle[u]:
						for w in rev_graph[u]:
							if w in comp_nodes and not in_cycle[w]:
								ways_w = (F[w][m] - F[w][v_val-1]) % mod
								if ways_w < 0:
									ways_w += mod
								product = (product * ways_w) % mod
				total_comp = (total_comp + product) % mod
			ans = (ans * total_comp) % mod
			
	print(ans)

if __name__ == '__main__':
	main()