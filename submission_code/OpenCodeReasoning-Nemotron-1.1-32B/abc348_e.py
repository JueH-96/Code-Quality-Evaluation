import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	index = 1
	graph = [[] for _ in range(n+1)]
	for i in range(n-1):
		a = int(data[index]); b = int(data[index+1]); index += 2
		graph[a].append(b)
		graph[b].append(a)
	
	C = list(map(int, data[index:index+n]))
	index += n
	total = sum(C)
	
	parent_arr = [-1] * (n+1)
	depth_arr = [-1] * (n+1)
	bfs_order = []
	queue = deque()
	queue.append(1)
	depth_arr[1] = 0
	parent_arr[1] = -1
	while queue:
		u = queue.popleft()
		bfs_order.append(u)
		for v in graph[u]:
			if v == parent_arr[u]:
				continue
			parent_arr[v] = u
			depth_arr[v] = depth_arr[u] + 1
			queue.append(v)
			
	order_desc = list(reversed(bfs_order))
	
	sum_subtree = [0] * (n+1)
	for node in order_desc:
		s = C[node-1]
		for neighbor in graph[node]:
			if neighbor == parent_arr[node]:
				continue
			s += sum_subtree[neighbor]
		sum_subtree[node] = s
		
	f1 = 0
	for i in range(1, n+1):
		f1 += depth_arr[i] * C[i-1]
		
	f = [0] * (n+1)
	f[1] = f1
	queue = deque()
	queue.append(1)
	while queue:
		u = queue.popleft()
		for v in graph[u]:
			if v == parent_arr[u]:
				continue
			f[v] = f[u] + total - 2 * sum_subtree[v]
			queue.append(v)
			
	ans = min(f[1:])
	print(ans)

if __name__ == "__main__":
	main()