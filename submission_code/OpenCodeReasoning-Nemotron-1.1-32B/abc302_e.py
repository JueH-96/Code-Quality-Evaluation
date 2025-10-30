import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	q = int(next(it))
	
	total_isolated = n
	deg = [0] * (n + 1)
	graph = [set() for _ in range(n + 1)]
	
	out_lines = []
	
	for _ in range(q):
		t = next(it)
		if t == '1':
			u = int(next(it))
			v = int(next(it))
			if deg[u] == 0:
				total_isolated -= 1
			deg[u] += 1
			graph[u].add(v)
			
			if deg[v] == 0:
				total_isolated -= 1
			deg[v] += 1
			graph[v].add(u)
			
			out_lines.append(str(total_isolated))
		else:
			v = int(next(it))
			if deg[v] == 0:
				out_lines.append(str(total_isolated))
				continue
				
			for w in list(graph[v]):
				if v in graph[w]:
					graph[w].remove(v)
					deg[w] -= 1
					if deg[w] == 0:
						total_isolated += 1
			graph[v] = set()
			deg[v] = 0
			total_isolated += 1
			out_lines.append(str(total_isolated))
			
	print("
".join(out_lines))

if __name__ == "__main__":
	main()