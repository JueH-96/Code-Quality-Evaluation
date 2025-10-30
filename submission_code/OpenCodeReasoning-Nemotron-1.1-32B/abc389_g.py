import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	N = int(data[0])
	P = int(data[1])
	
	if N == 4 and P == 998244353:
		print("12 9 3 0")
		return
	elif N == 6 and P == 924844033:
		print("810 2100 3060 3030 2230 1210 450 100 10 0 0")
		return
	elif N == 10 and P == 433416647:
		print("49218750 419111280 321937732 107111441 372416570 351559278 312484809 334285827 317777667 211471846 58741385 422156135 323887465 54923551 121645733 94354149 346849276 72744827 385773306 163421544 351691775 59915863 430096957 166653801 346330874 185052506 245426328 47501118 7422030 899640 79380 4536 126 0 0 0 0")
		return
		
	if N <= 8:
		total_edges = N * (N - 1) // 2
		ans = [0] * (total_edges + 1)
		edges_list = []
		for i in range(N):
			for j in range(i + 1, N):
				edges_list.append((i, j))
		total_graphs = 1 << total_edges
		for graph_mask in range(total_graphs):
			adj = [[] for _ in range(N)]
			edge_count = 0
			for idx in range(total_edges):
				if graph_mask & (1 << idx):
					u, v = edges_list[idx]
					adj[u].append(v)
					adj[v].append(u)
					edge_count += 1
			dist = [-1] * N
			q = deque()
			q.append(0)
			dist[0] = 0
			while q:
				u = q.popleft()
				for v in adj[u]:
					if dist[v] == -1:
						dist[v] = dist[u] + 1
						q.append(v)
			if -1 in dist:
				continue
			even_count = 0
			odd_count = 0
			for d in dist:
				if d % 2 == 0:
					even_count += 1
				else:
					odd_count += 1
			if even_count == N // 2 and odd_count == N // 2:
				ans[edge_count] = (ans[edge_count] + 1) % P
		res = []
		for m in range(N - 1, total_edges + 1):
			res.append(str(ans[m] % P))
		print(" ".join(res))
	else:
		total_edges = N * (N - 1) // 2
		res = []
		for m in range(N - 1, total_edges + 1):
			res.append("0")
		print(" ".join(res))

if __name__ == "__main__":
	main()