import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	H = int(next(it))
	W = int(next(it))
	grid = []
	for i in range(H):
		row = []
		for j in range(W):
			row.append(int(next(it)))
		grid.append(row)
	
	Q = int(next(it))
	queries = []
	queries_here = [[[] for _ in range(W)] for _ in range(H)]
	
	for idx in range(Q):
		a = int(next(it)); b = int(next(it)); y = int(next(it))
		c = int(next(it)); d = int(next(it)); z = int(next(it))
		a0 = a-1; b0 = b-1; c0 = c-1; d0 = d-1
		queries.append((a0, b0, c0, d0, y, z))
		queries_here[a0][b0].append(idx)
		queries_here[c0][d0].append(idx)
	
	total_nodes = H * W
	parent = list(range(total_nodes))
	size = [1] * total_nodes
	comp_queries = [set() for _ in range(total_nodes)]
	active = [[False] * W for _ in range(H)]
	
	def find(x):
		if parent[x] != x:
			parent[x] = find(parent[x])
		return parent[x]
	
	def union(x, y):
		x = find(x)
		y = find(y)
		if x == y:
			return x
		if size[x] < size[y]:
			x, y = y, x
		parent[y] = x
		size[x] += size[y]
		comp_queries[x] |= comp_queries[y]
		comp_queries[y] = set()
		return x

	buildings = []
	for i in range(H):
		for j in range(W):
			buildings.append((grid[i][j], i, j))
	buildings.sort(key=lambda x: x[0], reverse=True)
	
	ans_bottleneck = [-1] * Q
	directions = [(0,1), (0,-1), (1,0), (-1,0)]
	
	for f, i, j in buildings:
		idx = i * W + j
		active[i][j] = True
		root = find(idx)
		
		for q_index in queries_here[i][j]:
			comp_queries[root].add(q_index)
			
		for di, dj in directions:
			ni = i + di
			nj = j + dj
			if ni < 0 or ni >= H or nj < 0 or nj >= W:
				continue
			if not active[ni][nj]:
				continue
			nidx = ni * W + nj
			nroot = find(nidx)
			if root == nroot:
				continue
			new_root = union(root, nroot)
			root = new_root
			for q_index in comp_queries[root]:
				if ans_bottleneck[q_index] != -1:
					continue
				a1, b1, a2, b2, y, z = queries[q_index]
				if not (active[a1][b1] and active[a2][b2]):
					continue
				idx1 = a1 * W + b1
				idx2 = a2 * W + b2
				r1 = find(idx1)
				r2 = find(idx2)
				if r1 == r2:
					ans_bottleneck[q_index] = f
					
	for i in range(Q):
		a1, b1, a2, b2, y, z = queries[i]
		M0 = ans_bottleneck[i]
		if M0 >= min(y, z):
			res = abs(y - z)
		else:
			res = y + z - 2 * M0
		print(res)

if __name__ == "__main__":
	main()