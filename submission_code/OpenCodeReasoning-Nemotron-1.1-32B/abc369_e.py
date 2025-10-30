import sys
import itertools

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	
	n = int(next(it)); m = int(next(it))
	INF = 10**18
	dist = [[INF] * (n+1) for _ in range(n+1)]
	for i in range(1, n+1):
		dist[i][i] = 0
		
	bridges = [None] * (m+1)
	
	for i in range(1, m+1):
		u = int(next(it)); v = int(next(it)); t = int(next(it))
		bridges[i] = (u, v, t)
		if t < dist[u][v]:
			dist[u][v] = t
			dist[v][u] = t
			
	for k in range(1, n+1):
		for i in range(1, n+1):
			if dist[i][k] == INF:
				continue
			for j in range(1, n+1):
				new_dist = dist[i][k] + dist[k][j]
				if new_dist < dist[i][j]:
					dist[i][j] = new_dist
					
	q = int(next(it))
	out_lines = []
	for _ in range(q):
		k = int(next(it))
		bridge_list = [int(next(it)) for _ in range(k)]
		forced_edges = []
		total_forced_cost = 0
		for b in bridge_list:
			u, v, t = bridges[b]
			forced_edges.append((u, v, t))
			total_forced_cost += t
			
		best = INF
		for perm in itertools.permutations(forced_edges):
			for mask in range(1 << k):
				current = 1
				extra = 0
				valid = True
				for i in range(k):
					u, v, t_val = perm[i]
					if mask & (1 << i):
						start = v
						end = u
					else:
						start = u
						end = v
					if dist[current][start] == INF:
						valid = False
						break
					extra += dist[current][start]
					current = end
					
				if not valid:
					continue
					
				if dist[current][n] == INF:
					continue
				extra += dist[current][n]
				total_cost = total_forced_cost + extra
				if total_cost < best:
					best = total_cost
					
		out_lines.append(str(best))
		
	print("
".join(out_lines))

if __name__ == "__main__":
	main()