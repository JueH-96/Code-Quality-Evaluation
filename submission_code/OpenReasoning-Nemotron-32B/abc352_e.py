import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(-1)
		return
	
	n = int(data[0])
	m = int(data[1])
	index = 2
	operations = []
	for i in range(m):
		k = int(data[index])
		c = int(data[index+1])
		index += 2
		arr = list(map(int, data[index:index+k]))
		index += k
		operations.append((k, c, arr))
	
	operations.sort(key=lambda x: x[1])
	
	parent = list(range(n+1))
	rank = [0] * (n+1)
	comp_count = n
	
	def find(x):
		if parent[x] != x:
			parent[x] = find(parent[x])
		return parent[x]
	
	def union(x, y):
		nonlocal comp_count
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
		comp_count -= 1
	
	total_cost = 0
	for op in operations:
		k, c, arr = op
		roots = set()
		for node in arr:
			roots.add(find(node))
		t = len(roots)
		if t <= 1:
			continue
		total_cost += (t-1) * c
		base = roots.pop()
		for r in roots:
			union(base, r)
	
	if comp_count == 1:
		print(total_cost)
	else:
		print(-1)

if __name__ == '__main__':
	main()