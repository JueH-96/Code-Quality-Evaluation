import sys
import heapq
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		print(-1)
		return
	n = int(data[0])
	edges = []
	graph = [[] for _ in range(n)]
	index = 1
	for i in range(n-1):
		u = int(data[index]) - 1
		v = int(data[index+1]) - 1
		index += 2
		graph[u].append(v)
		graph[v].append(u)
	
	deg = [len(graph[i]) for i in range(n)]
	if max(deg) < 4:
		print(-1)
		return

	parent = [-1] * n
	children = [[] for _ in range(n)]
	stack = [0]
	while stack:
		u = stack.pop()
		for v in graph[u]:
			if v == parent[u]:
				continue
			parent[v] = u
			children[u].append(v)
			stack.append(v)
	
	order = []
	stack = [0]
	while stack:
		u = stack.pop()
		order.append(u)
		for v in children[u]:
			stack.append(v)
	order.reverse()
	
	f = [-10**18] * n
	h_node = [0] * n
	
	for u in order:
		values = []
		for v in children[u]:
			values.append(h_node[v])
		if len(values) < 3:
			f[u] = -10**18
		else:
			values.sort(reverse=True)
			top3 = values[:3]
			f[u] = 1 + sum(top3)
		h_node[u] = max(1, f[u])
	
	parent_contribution = [0] * n
	q = deque([0])
	while q:
		p = q.popleft()
		child_list = children[p]
		L = [h_node[v] for v in child_list]
		count = len(L)
		if count < 3:
			for v in child_list:
				parent_contribution[v] = 1
		else:
			top4 = heapq.nlargest(4, L)
			top3 = top4[:3]
			top3_sum = sum(top3)
			top4_sum = sum(top4)
			if count == 3:
				for v in child_list:
					parent_contribution[v] = 1
			else:
				for v in child_list:
					x = h_node[v]
					if x in top4:
						new_top3_sum = top4_sum - x
						parent_contribution[v] = 1 + new_top3_sum
					else:
						parent_contribution[v] = 1 + top3_sum
		for v in child_list:
			q.append(v)
			
	total_arr = [-10**18] * n
	for u in range(n):
		branches = []
		if u != 0:
			branches.append(parent_contribution[u])
		for v in children[u]:
			branches.append(h_node[v])
		if len(branches) < 4:
			continue
		top4_branches = heapq.nlargest(4, branches)
		total_arr[u] = 1 + sum(top4_branches)
	
	ans = max(total_arr)
	if ans < 5:
		print(-1)
	else:
		print(ans)

if __name__ == "__main__":
	main()