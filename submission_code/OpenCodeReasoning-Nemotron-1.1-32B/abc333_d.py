import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	graph = [[] for _ in range(n+1)]
	index = 1
	for i in range(n-1):
		u = int(data[index])
		v = int(data[index+1])
		index += 2
		graph[u].append(v)
		graph[v].append(u)
	
	parent = [-1] * (n+1)
	size = [1] * (n+1)
	
	stack = [1]
	order = []
	while stack:
		u = stack.pop()
		order.append(u)
		for v in graph[u]:
			if v == parent[u]:
				continue
			parent[v] = u
			stack.append(v)
	
	for i in range(len(order)-1, -1, -1):
		u = order[i]
		if u != 1:
			size[parent[u]] += size[u]
	
	children = graph[1]
	if children:
		max_child_size = max(size[child] for child in children)
	else:
		max_child_size = 0
		
	ans = n - max_child_size
	print(ans)

if __name__ == "__main__":
	main()