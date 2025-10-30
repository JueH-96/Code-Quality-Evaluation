import sys

def main():
	data = sys.stdin.read().split()
	if not data: 
		return
	
	it = iter(data)
	n = int(next(it)); m = int(next(it)); q = int(next(it))
	
	g = [[] for _ in range(n)]
	roads = []
	for i in range(m):
		a = int(next(it)) - 1
		b = int(next(it)) - 1
		c = int(next(it))
		roads.append((a, b, c))
		g[a].append((b, c, i))
		g[b].append((a, c, i))
	
	closed_set = set()
	out_lines = []
	INF = 10**18
	
	for _ in range(q):
		t = next(it)
		if t == '1':
			idx = int(next(it)) - 1
			closed_set.add(idx)
		else:
			x = int(next(it)) - 1
			y = int(next(it)) - 1
			dist = [INF] * n
			visited = [False] * n
			dist[x] = 0
			
			for _ in range(n):
				u = -1
				min_dist = INF
				for i in range(n):
					if not visited[i] and dist[i] < min_dist:
						min_dist = dist[i]
						u = i
				if u == -1:
					break
				if u == y:
					break
				visited[u] = True
				for v, w, idx_road in g[u]:
					if idx_road in closed_set:
						continue
					if not visited[v]:
						new_dist = dist[u] + w
						if new_dist < dist[v]:
							dist[v] = new_dist
			if dist[y] == INF:
				out_lines.append("-1")
			else:
				out_lines.append(str(dist[y]))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()