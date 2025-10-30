import sys
import bisect
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	n = int(next(it)); m = int(next(it)); q = int(next(it))
	s_list = []
	t_list = []
	for _ in range(m):
		s_val = int(next(it)); t_val = int(next(it))
		s_list.append(s_val)
		t_list.append(t_val)
	
	queries = []
	for _ in range(q):
		L = int(next(it)); R = int(next(it))
		queries.append((L, R))
		
	out_lines = []
	for (L, R) in queries:
		V = set()
		for i in range(L-1, R):
			V.add(s_list[i])
			V.add(t_list[i])
		if not V:
			out_lines.append("Yes")
			continue
			
		parent = {}
		rank = {}
		for x in V:
			parent[x] = x
			rank[x] = 0
			
		def find(x):
			if parent[x] != x:
				parent[x] = find(parent[x])
			return parent[x]
		
		def union(x, y):
			x = find(x)
			y = find(y)
			if x == y:
				return
			if rank[x] < rank[y]:
				parent[x] = y
			elif rank[x] > rank[y]:
				parent[y] = x
			else:
				parent[y] = x
				rank[x] += 1
				
		for i in range(L-1, R):
			a = s_list[i]
			b = t_list[i]
			if a not in parent or b not in parent:
				continue
			union(a, b)
			
		group_of = {}
		for x in V:
			group_of[x] = find(x)
			
		sorted_V = sorted(V)
		graph = {}
		groups_set = set(group_of.values())
		
		for i in range(L-1, R):
			a_val = s_list[i]
			b_val = t_list[i]
			a = min(a_val, b_val)
			b = max(a_val, b_val)
			c = group_of[a]
			
			left_index = bisect.bisect_right(sorted_V, a)
			right_index = bisect.bisect_left(sorted_V, b) - 1
			if left_index > right_index:
				continue
				
			distinct_groups = set()
			for j in range(left_index, right_index+1):
				town = sorted_V[j]
				d = group_of[town]
				distinct_groups.add(d)
				
			for d in distinct_groups:
				if a_val < b_val:
					if c not in graph:
						graph[c] = set()
					if d not in graph[c]:
						graph[c].add(d)
				else:
					if d not in graph:
						graph[d] = set()
					if c not in graph[d]:
						graph[d].add(c)
						
		all_groups = groups_set
		in_degree = {g: 0 for g in all_groups}
		for u in graph:
			for v in graph[u]:
				in_degree[v] += 1
				
		q_kahn = deque()
		for g in all_groups:
			if in_degree[g] == 0:
				q_kahn.append(g)
				
		order = []
		while q_kahn:
			u = q_kahn.popleft()
			order.append(u)
			if u in graph:
				for v in graph[u]:
					in_degree[v] -= 1
					if in_degree[v] == 0:
						q_kahn.append(v)
						
		if len(order) == len(all_groups):
			out_lines.append("Yes")
		else:
			out_lines.append("No")
			
	for line in out_lines:
		print(line)

if __name__ == '__main__':
	main()