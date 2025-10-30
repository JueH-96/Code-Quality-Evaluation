import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	n = int(next(it))
	q = int(next(it))
	
	parent = list(range(n+1))
	rank = [0] * (n+1)
	top10 = [[i] for i in range(n+1)]
	
	def find(x):
		path = []
		while parent[x] != x:
			path.append(x)
			x = parent[x]
		root = x
		for node in path:
			parent[node] = root
		return root

	out_lines = []
	for _ in range(q):
		t = next(it)
		if t == '1':
			u = int(next(it))
			v = int(next(it))
			ru = find(u)
			rv = find(v)
			if ru == rv:
				continue
			if rank[ru] < rank[rv]:
				ru, rv = rv, ru
			parent[rv] = ru
			if rank[ru] == rank[rv]:
				rank[ru] += 1
			combined = top10[ru] + top10[rv]
			combined.sort(reverse=True)
			top10[ru] = combined[:10]
		else:
			v = int(next(it))
			k = int(next(it))
			root = find(v)
			if k > len(top10[root]):
				out_lines.append("-1")
			else:
				out_lines.append(str(top10[root][k-1]))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()