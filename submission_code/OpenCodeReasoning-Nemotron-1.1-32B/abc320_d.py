import sys
from collections import deque

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	
	n, m = map(int, data[0].split())
	graph = [[] for _ in range(n)]
	
	for i in range(1, m + 1):
		parts = data[i].split()
		if not parts:
			continue
		a = int(parts[0]) - 1
		b = int(parts[1]) - 1
		x = int(parts[2])
		y = int(parts[3])
		graph[a].append((b, x, y))
		graph[b].append((a, -x, -y))
	
	coords = [None] * n
	if n > 0:
		coords[0] = (0, 0)
		queue = deque([0])
		visited_nodes = set([0])
		inconsistent = False
		
		while queue:
			u = queue.popleft()
			for (v, dx, dy) in graph[u]:
				nx = coords[u][0] + dx
				ny = coords[u][1] + dy
				if coords[v] is None:
					coords[v] = (nx, ny)
					visited_nodes.add(v)
					queue.append(v)
				else:
					if coords[v][0] != nx or coords[v][1] != ny:
						inconsistent = True
		
		if inconsistent:
			for node in visited_nodes:
				coords[node] = None
	
	for i in range(n):
		if coords[i] is None:
			print("undecidable")
		else:
			print(f"{coords[i][0]} {coords[i][1]}")

if __name__ == "__main__":
	main()