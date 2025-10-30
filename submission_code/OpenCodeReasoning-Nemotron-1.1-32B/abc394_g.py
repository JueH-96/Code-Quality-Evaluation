import sys
from collections import deque

def main():
	input = sys.stdin.readline
	H, W = map(int, input().split())
	F = []
	for _ in range(H):
		F.append(list(map(int, input().split())))
	
	n = H * W
	edges = []
	for i in range(H):
		for j in range(W):
			if j + 1 < W:
				w = min(F[i][j], F[i][j+1])
				edges.append((w, i, j, i, j+1))
			if i + 1 < H:
				w = min(F[i][j], F[i+1][j])
				edges.append((w, i, j, i+1, j))
	
	edges.sort(key=lambda x: -x[0])
	
	parent_dsu = list(range(n))
	rank = [0] * n
	
	def find(x):
		if parent_dsu[x] != x:
			parent_dsu[x] = find(parent_dsu[x])
		return parent_dsu[x]
	
	def union(x, y):
		rx = find(x)
		ry = find(y)
		if rx == ry:
			return False
		if rank[rx] < rank[ry]:
			parent_dsu[rx] = ry
		elif rank[rx] > rank[ry]:
			parent_dsu[ry] = rx
		else:
			parent_dsu[ry] = rx
			rank[rx] += 1
		return True

	graph = [[] for _ in range(n)]
	for edge in edges:
		w, i1, j1, i2, j2 = edge
		u = i1 * W + j1
		v = i2 * W + j2
		if union(u, v):
			graph[u].append((v, w))
			graph[v].append((u, w))
	
	depth = [-1] * n
	parent0 = [-1] * n
	min_edge0 = [0] * n
	q = deque([0])
	depth[0] = 0
	parent0[0] = -1
	min_edge0[0] = 10**18
	
	while q:
		u = q.popleft()
		for v, w in graph[u]:
			if v == parent0[u]:
				continue
			depth[v] = depth[u] + 1
			parent0[v] = u
			min_edge0[v] = w
			q.append(v)
	
	LOG = (n).bit_length()
	parent_table = [[-1] * n for _ in range(LOG)]
	min_edge_table = [[10**18] * n for _ in range(LOG)]
	
	for i in range(n):
		parent_table[0][i] = parent0[i]
		min_edge_table[0][i] = min_edge0[i]
	
	for k in range(1, LOG):
		for i in range(n):
			p = parent_table[k-1][i]
			if p == -1:
				parent_table[k][i] = -1
				min_edge_table[k][i] = min_edge_table[k-1][i]
			else:
				parent_table[k][i] = parent_table[k-1][p]
				min_edge_table[k][i] = min(min_edge_table[k-1][i], min_edge_table[k-1][p])
	
	Q = int(input().strip())
	res = []
	for _ in range(Q):
		A, B, Y, C, D, Z = map(int, input().split())
		u = (A-1) * W + (B-1)
		v = (C-1) * W + (D-1)
		
		min_val = 10**18
		if depth[u] < depth[v]:
			u, v = v, u
		
		d = depth[u] - depth[v]
		bit = 0
		while d:
			if d & 1:
				if u == -1:
					break
				min_val = min(min_val, min_edge_table[bit][u])
				u = parent_table[bit][u]
			d //= 2
			bit += 1
		
		if u == v:
			M0 = min_val
		else:
			for k in range(LOG-1, -1, -1):
				if parent_table[k][u] != parent_table[k][v]:
					min_val = min(min_val, min_edge_table[k][u], min_edge_table[k][v])
					u = parent_table[k][u]
					v = parent_table[k][v]
			min_val = min(min_val, min_edge_table[0][u], min_edge_table[0][v])
			M0 = min_val
		
		if M0 >= min(Y, Z):
			ans = abs(Y - Z)
		else:
			ans = Y + Z - 2 * M0
		res.append(str(ans))
	
	print("
".join(res))

if __name__ == "__main__":
	main()