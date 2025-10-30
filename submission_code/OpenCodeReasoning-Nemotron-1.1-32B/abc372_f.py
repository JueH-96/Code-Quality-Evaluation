mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	N = int(next(it)); M = int(next(it)); K = int(next(it))
	edges = []
	for i in range(M):
		x = int(next(it)); y = int(next(it))
		edges.append((x, y))
	
	V0_set = set([1])
	for (x, y) in edges:
		V0_set.add(x)
		V0_set.add(y)
	V0 = sorted(V0_set)
	L = len(V0)
	vertex_to_index = {v: i for i, v in enumerate(V0)}
	
	base_next = [0] * L
	base_weight = [0] * L
	for i, v in enumerate(V0):
		min_dist = 10**18
		next_v = None
		for w in V0:
			if w == v:
				continue
			if w > v:
				d = w - v
			else:
				d = N - v + w
			if d < min_dist:
				min_dist = d
				next_v = w
		if next_v is None:
			next_v = v
			min_dist = N
		base_next[i] = next_v
		base_weight[i] = min_dist

	extra_in = [[] for _ in range(L)]
	base_in = [[] for _ in range(L)]
	
	for (x, y) in edges:
		i_x = vertex_to_index[x]
		i_y = vertex_to_index[y]
		extra_in[i_y].append(i_x)
	
	for i in range(L):
		v = base_next[i]
		j = vertex_to_index[v]
		base_in[j].append(i)
	
	dp = [[0] * L for _ in range(K+1)]
	if 1 in vertex_to_index:
		idx0 = vertex_to_index[1]
		dp[0][idx0] = 1
	else:
		dp[0] = [0] * L
	
	for k in range(1, K+1):
		for i in range(L):
			total = 0
			for j in extra_in[i]:
				if k-1 >= 0:
					total = (total + dp[k-1][j]) % mod
			for j in base_in[i]:
				w = base_weight[j]
				if k - w >= 0:
					total = (total + dp[k-w][j]) % mod
			dp[k][i] = total

	ans = 0
	for t in range(0, K+1):
		for i in range(L):
			if t == K:
				ans = (ans + dp[t][i]) % mod
			else:
				r = K - t
				if r < base_weight[i]:
					ans = (ans + dp[t][i]) % mod
	print(ans)

if __name__ == "__main__":
	main()