import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	q = int(next(it))
	
	roads = []
	for _ in range(m):
		a = int(next(it)) - 1
		b = int(next(it)) - 1
		c = int(next(it))
		roads.append((a, b, c))
	
	INF = 10**18
	dist = [[INF] * n for _ in range(n)]
	for i in range(n):
		dist[i][i] = 0
		
	for a, b, c in roads:
		if c < dist[a][b]:
			dist[a][b] = c
			dist[b][a] = c
			
	for k in range(n):
		for i in range(n):
			if dist[i][k] == INF:
				continue
			for j in range(n):
				if dist[k][j] == INF:
					continue
				if dist[i][j] > dist[i][k] + dist[k][j]:
					dist[i][j] = dist[i][k] + dist[k][j]
					
	out_lines = []
	for _ in range(q):
		t = next(it)
		if t == '1':
			i = int(next(it)) - 1
			a, b, c = roads[i]
			dist[a][b] = INF
			dist[b][a] = INF
			for k in range(n):
				for i in range(n):
					if dist[i][k] == INF:
						continue
					for j in range(n):
						if dist[k][j] == INF:
							continue
						if dist[i][j] > dist[i][k] + dist[k][j]:
							dist[i][j] = dist[i][k] + dist[k][j]
		else:
			x = int(next(it)) - 1
			y = int(next(it)) - 1
			if dist[x][y] == INF:
				out_lines.append("-1")
			else:
				out_lines.append(str(dist[x][y]))
				
	print("
".join(out_lines))

if __name__ == "__main__":
	main()