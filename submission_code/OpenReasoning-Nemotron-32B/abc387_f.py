import sys
from collections import deque, defaultdict

mod = 998244353

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+n]))
	nxt = [a-1 for a in A]

	children = [[] for _ in range(n)]
	in_degree = [0] * n
	for i in range(n):
		parent_i = nxt[i]
		children[parent_i].append(i)
		in_degree[parent_i] += 1

	temp_degree = in_degree[:]
	q = deque()
	for i in range(n):
		if temp_degree[i] == 0:
			q.append(i)
			
	in_cycle = [False] * n
	while q:
		u = q.popleft()
		v = nxt[u]
		temp_degree[v] -= 1
		if temp_degree[v] == 0:
			q.append(v)
			
	for i in range(n):
		if temp_degree[i] > 0:
			in_cycle[i] = True

	comp_root = [-1] * n
	visited_cycle = [False] * n
	for i in range(n):
		if in_cycle[i] and not visited_cycle[i]:
			cycle_nodes = []
			cur = i
			while not visited_cycle[cur]:
				visited_cycle[cur] = True
				cycle_nodes.append(cur)
				cur = nxt[cur]
			rep = min(cycle_nodes)
			for node in cycle_nodes:
				comp_root[node] = rep
				
	q = deque()
	for i in range(n):
		if in_cycle[i]:
			q.append(i)
	while q:
		u = q.popleft()
		for child in children[u]:
			if not in_cycle[child]:
				comp_root[child] = comp_root[u]
				q.append(child)
				
	tree_children = []
	for u in range(n):
		t_children = []
		for child in children[u]:
			if not in_cycle[child]:
				t_children.append(child)
		tree_children.append(t_children)
		
	dp = [[0] * (m+1) for _ in range(n)]
	deg = [len(tree_children[u]) for u in range(n)]
	
	q = deque()
	for u in range(n):
		if deg[u] == 0:
			q.append(u)
			
	while q:
		u = q.popleft()
		arr = [1] * (m+1)
		
		for c in tree_children[u]:
			prefix = [0] * (m+1)
			for w in range(1, m+1):
				prefix[w] = (prefix[w-1] + dp[c][w]) % mod
			for w in range(1, m+1):
				arr[w] = (arr[w] * prefix[w]) % mod
				
		dp[u] = arr
		
		if not in_cycle[u]:
			p = nxt[u]
			deg[p] -= 1
			if deg[p] == 0:
				q.append(p)
				
	comp_dict = defaultdict(list)
	for u in range(n):
		if in_cycle[u]:
			rep = comp_root[u]
			comp_dict[rep].append(u)
			
	ans = 1
	for rep, cycle_list in comp_dict.items():
		total_comp = 0
		for v in range(1, m+1):
			prod = 1
			for u in cycle_list:
				prod = (prod * dp[u][v]) % mod
			total_comp = (total_comp + prod) % mod
		ans = (ans * total_comp) % mod
		
	print(ans)

if __name__ == '__main__':
	main()