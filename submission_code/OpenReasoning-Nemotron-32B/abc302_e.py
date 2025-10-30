import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	q = int(next(it))
	deg = [0] * (n + 1)
	adj = [set() for _ in range(n + 1)]
	isolated_count = n
	out_lines = []

	for _ in range(q):
		t = int(next(it))
		if t == 1:
			u = int(next(it))
			v = int(next(it))
			if deg[u] == 0:
				isolated_count -= 1
			if deg[v] == 0:
				isolated_count -= 1
			adj[u].add(v)
			adj[v].add(u)
			deg[u] += 1
			deg[v] += 1
			out_lines.append(str(isolated_count))
		else:
			v = int(next(it))
			if deg[v] > 0:
				while adj[v]:
					w = adj[v].pop()
					adj[w].remove(v)
					deg[w] -= 1
					if deg[w] == 0:
						isolated_count += 1
				deg[v] = 0
				isolated_count += 1
			out_lines.append(str(isolated_count))
	
	print("
".join(out_lines))

if __name__ == '__main__':
	main()