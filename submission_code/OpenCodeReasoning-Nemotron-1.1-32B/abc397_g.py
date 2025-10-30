import collections
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	m = int(data[1])
	k = int(data[2])
	edges = []
	out_edges = [[] for _ in range(n+1)]
	in_edges = [[] for _ in range(n+1)]
	index = 3
	for i in range(m):
		u = int(data[index])
		v = int(data[index+1])
		index += 2
		edges.append((u, v))
		out_edges[u].append(v)
		in_edges[v].append(u)
	
	dist = [-1] * (n+1)
	q = collections.deque()
	dist[1] = 0
	q.append(1)
	while q:
		u = q.popleft()
		for v in out_edges[u]:
			if dist[v] == -1:
				dist[v] = dist[u] + 1
				q.append(v)
	L0 = dist[n]
	
	inf = 10**9
	for d_star in range(L0, -1, -1):
		dp_prev = [inf] * (n+1)
		for c in range(0, d_star+1):
			dp_curr = [inf] * (n+1)
			if c == 0:
				dp_curr[1] = 0
			else:
				dp_curr[1] = inf
				
			for _ in range(n):
				changed = False
				for v in range(1, n+1):
					total = 0
					for u in in_edges[v]:
						if c - 1 >= 0:
							option1 = dp_prev[u] + 1
						else:
							option1 = 1
						option2 = dp_curr[u]
						cost_edge = min(option1, option2)
						total += cost_edge
					if total < dp_curr[v]:
						dp_curr[v] = total
						changed = True
				if not changed:
					break
			dp_prev = dp_curr
		F_val = dp_curr[n]
		if F_val <= k:
			print(d_star)
			return
	print(0)

if __name__ == '__main__':
	main()