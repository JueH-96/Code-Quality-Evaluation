import collections

def main():
	n = int(input().strip())
	graph = [[] for _ in range(n+1)]
	for _ in range(n-1):
		u, v = map(int, input().split())
		graph[u].append(v)
		graph[v].append(u)
	
	parent = [0] * (n+1)
	size = [0] * (n+1)
	
	q = collections.deque([1])
	parent[1] = 0
	order = []
	while q:
		u = q.popleft()
		order.append(u)
		for v in graph[u]:
			if v == parent[u]:
				continue
			parent[v] = u
			q.append(v)
			
	for i in range(len(order)-1, -1, -1):
		u = order[i]
		size[u] = 1
		for v in graph[u]:
			if v == parent[u]:
				continue
			size[u] += size[v]
			
	children = []
	for v in graph[1]:
		if v != parent[1]:
			children.append(v)
			
	max_size = max(size[v] for v in children) if children else 0
	ans = n - max_size
	print(ans)

if __name__ == '__main__':
	main()