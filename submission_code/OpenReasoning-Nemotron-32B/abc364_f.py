import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	q = int(data[1])
	queries = []
	index = 2
	for i in range(q):
		l = int(data[index])
		r = int(data[index+1])
		c = int(data[index+2])
		index += 3
		queries.append((l-1, r-1, c, i))
	
	queries.sort(key=lambda x: x[2])
	total_vertices = n + q
	par = list(range(total_vertices))
	nxt = [i + 1 for i in range(n)]
	if n > 0:
		nxt[n-1] = total_vertices
	else:
		nxt = []
	
	ans = 0
	for l, r, c, i in queries:
		hub = n + i
		cur = l
		while cur <= r:
			root = cur
			stack = []
			while par[root] != root:
				stack.append(root)
				root = par[root]
			for node in stack:
				par[node] = root
			if root > r:
				break
				
			hub_root = hub
			stack = []
			while par[hub_root] != hub_root:
				stack.append(hub_root)
				hub_root = par[hub_root]
			for node in stack:
				par[node] = hub_root
				
			if root != hub_root:
				ans += c
				if hub_root < root:
					par[root] = hub_root
				else:
					par[hub_root] = root
					hub_root = root
					
			if nxt[root] <= r:
				next_node = nxt[root]
				stack = []
				while par[next_node] != next_node:
					stack.append(next_node)
					next_node = par[next_node]
				for node in stack:
					par[node] = next_node
				if next_node != hub_root:
					if next_node < hub_root:
						par[hub_root] = next_node
					else:
						par[next_node] = hub_root
				nxt[root] = nxt[next_node]
			cur = nxt[root]
	
	components = set()
	for i in range(n):
		root = i
		while par[root] != root:
			root = par[root]
		components.add(root)
	if len(components) > 1:
		print(-1)
	else:
		print(ans)

if __name__ == "__main__":
	main()