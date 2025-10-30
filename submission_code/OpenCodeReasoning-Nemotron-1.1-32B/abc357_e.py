import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	a = list(map(int, data[1:1+n]))
	next_arr = [x-1 for x in a]
	
	visited = [0] * n
	in_cycle = [False] * n
	cycle_length_arr = [0] * n
	
	for i in range(n):
		if visited[i]:
			continue
		stack = []
		pos_in_stack = {}
		cur = i
		while not visited[cur]:
			visited[cur] = 1
			stack.append(cur)
			pos_in_stack[cur] = len(stack) - 1
			cur = next_arr[cur]
		
		if cur in pos_in_stack:
			idx = pos_in_stack[cur]
			cycle_nodes = stack[idx:]
			L = len(cycle_nodes)
			for node in cycle_nodes:
				in_cycle[node] = True
				cycle_length_arr[node] = L
				
	rev_graph = [[] for _ in range(n)]
	for u in range(n):
		v = next_arr[u]
		rev_graph[v].append(u)
		
	depth = [-1] * n
	q = deque()
	for i in range(n):
		if in_cycle[i]:
			depth[i] = 0
			q.append(i)
			
	while q:
		u = q.popleft()
		for v in rev_graph[u]:
			if not in_cycle[v] and depth[v] == -1:
				depth[v] = depth[u] + 1
				cycle_length_arr[v] = cycle_length_arr[u]
				q.append(v)
				
	total = 0
	for i in range(n):
		if in_cycle[i]:
			total += cycle_length_arr[i]
		else:
			total += depth[i] + cycle_length_arr[i]
			
	print(total)

if __name__ == "__main__":
	main()