import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	m = int(data[1])
	edges = []
	idx = 2
	for i in range(m):
		u = int(data[idx])
		v = int(data[idx + 1])
		idx += 2
		edges.append((u - 1, v - 1))
	
	parent = list(range(n))
	rank = [0] * n
	
	def find(x):
		if parent[x] != x:
			parent[x] = find(parent[x])
		return parent[x]
	
	def union(x, y):
		rx = find(x)
		ry = find(y)
		if rx == ry:
			return
		if rank[rx] < rank[ry]:
			parent[rx] = ry
		elif rank[rx] > rank[ry]:
			parent[ry] = rx
		else:
			parent[ry] = rx
			rank[rx] += 1
			
	for u, v in edges:
		union(u, v)
		
	comp_nodes = {}
	comp_edges = {}
	
	for i in range(n):
		r = find(i)
		comp_nodes[r] = comp_nodes.get(r, 0) + 1
		
	for u, v in edges:
		r = find(u)
		comp_edges[r] = comp_edges.get(r, 0) + 1
		
	ans = 0
	for r in comp_nodes:
		n_i = comp_nodes[r]
		m_i = comp_edges.get(r, 0)
		total_edges = n_i * (n_i - 1) // 2
		ans += total_edges - m_i
		
	print(ans)

if __name__ == '__main__':
	main()