import heapq
MOD1 = 10**9 + 7
MOD2 = 10**9 + 9

def main():
	import sys
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	graph = [[] for _ in range(n)]
	edges = []
	for _ in range(m):
		a = int(next(it)) - 1
		b = int(next(it)) - 1
		c = int(next(it))
		graph[a].append((b, c))
		graph[b].append((a, c))
		edges.append((a, b, c))
	
	dist1 = [10**18] * n
	dist1[0] = 0
	heap = [(0, 0)]
	while heap:
		d, u = heapq.heappop(heap)
		if d != dist1[u]:
			continue
		for v, c in graph[u]:
			nd = d + c
			if nd < dist1[v]:
				dist1[v] = nd
				heapq.heappush(heap, (nd, v))
	
	distN = [10**18] * n
	distN[n - 1] = 0
	heap = [(0, n - 1)]
	while heap:
		d, u = heapq.heappop(heap)
		if d != distN[u]:
			continue
		for v, c in graph[u]:
			nd = d + c
			if nd < distN[v]:
				distN[v] = nd
				heapq.heappush(heap, (nd, v))
	
	d0 = dist1[n - 1]
	
	dp1_mod1 = [0] * n
	dp1_mod2 = [0] * n
	dp1_mod1[0] = 1
	dp1_mod2[0] = 1
	nodes = sorted(range(n), key=lambda x: dist1[x])
	for u in nodes:
		for v, c in graph[u]:
			if dist1[u] + c == dist1[v]:
				dp1_mod1[v] = (dp1_mod1[v] + dp1_mod1[u]) % MOD1
				dp1_mod2[v] = (dp1_mod2[v] + dp1_mod2[u]) % MOD2
	
	dp2_mod1 = [0] * n
	dp2_mod2 = [0] * n
	dp2_mod1[n - 1] = 1
	dp2_mod2[n - 1] = 1
	nodes2 = sorted(range(n), key=lambda x: distN[x])
	for u in nodes2:
		for v, c in graph[u]:
			if distN[u] == distN[v] + c:
				dp2_mod1[u] = (dp2_mod1[u] + dp2_mod1[v]) % MOD1
				dp2_mod2[u] = (dp2_mod2[u] + dp2_mod2[v]) % MOD2
	
	total1 = dp1_mod1[n - 1]
	total2 = dp1_mod2[n - 1]
	
	out_lines = []
	for u, v, c in edges:
		count1 = 0
		count2 = 0
		if dist1[u] + c + distN[v] == d0:
			count1 = (count1 + dp1_mod1[u] * dp2_mod1[v]) % MOD1
			count2 = (count2 + dp1_mod2[u] * dp2_mod2[v]) % MOD2
		if dist1[v] + c + distN[u] == d0:
			count1 = (count1 + dp1_mod1[v] * dp2_mod1[u]) % MOD1
			count2 = (count2 + dp1_mod2[v] * dp2_mod2[u]) % MOD2
		if count1 == total1 and count2 == total2:
			out_lines.append("Yes")
		else:
			out_lines.append("No")
	
	print("
".join(out_lines))

if __name__ == '__main__':
	main()