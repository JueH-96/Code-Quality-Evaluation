import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	operations = []
	for _ in range(m):
		k = int(next(it))
		c = int(next(it))
		arr = [int(next(it)) for _ in range(k)]
		operations.append((k, c, arr))
	
	operations.sort(key=lambda x: x[1])
	
	parent = list(range(n + 1))
	rank = [0] * (n + 1)
	
	def find(x):
		root = x
		while root != parent[root]:
			root = parent[root]
		while x != root:
			next_node = parent[x]
			parent[x] = root
			x = next_node
		return root
	
	def union(x, y):
		rx = find(x)
		ry = find(y)
		if rx == ry:
			return False
		if rank[rx] < rank[ry]:
			parent[rx] = ry
		elif rank[rx] > rank[ry]:
			parent[ry] = rx
		else:
			parent[ry] = rx
			rank[rx] += 1
		return True

	total_cost = 0
	for k, c, arr in operations:
		comp_set = set()
		for v in arr:
			comp_set.add(find(v))
		comp_list = list(comp_set)
		k_comp = len(comp_list)
		if k_comp > 1:
			base = comp_list[0]
			for i in range(1, k_comp):
				union(base, comp_list[i])
			total_cost += (k_comp - 1) * c
			
	comp_set_final = set()
	for i in range(1, n + 1):
		comp_set_final.add(find(i))
	if len(comp_set_final) == 1:
		print(total_cost)
	else:
		print(-1)

if __name__ == '__main__':
	main()