mod = 998244353

import sys
import bisect

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
	
	if M == 0:
		print(1)
		return

	S_set = set([1])
	for (x, y) in edges:
		S_set.add(x)
		S_set.add(y)
	S = sorted(S_set)
	idx_map = {}
	for idx, vertex in enumerate(S):
		idx_map[vertex] = idx
	n_S = len(S)
	
	next_base = [0] * (N+1)
	for i in range(1, N):
		next_base[i] = i+1
	next_base[N] = 1

	nxt = [None] * (N+1)
	B = sorted(S)
	for i in range(1, N+1):
		if i in idx_map:
			nxt[i] = (i, 0)
		else:
			pos = bisect.bisect_right(B, i)
			if pos < len(B):
				j = B[pos]
				nxt[i] = (j, j - i)
			else:
				j = B[0]
				nxt[i] = (j, (N - i) + j)
				
	extra_list = {}
	for (x,y) in edges:
		if x not in extra_list:
			extra_list[x] = []
		extra_list[x].append(y)
		
	trans = {}
	for v in S:
		trans[v] = []
		w_base = next_base[v]
		u_base, d_base = nxt[w_base]
		trans[v].append( (u_base, d_base) )
		if v in extra_list:
			for w in extra_list[v]:
				u, d = nxt[w]
				trans[v].append( (u, d) )
				
	dp = [[0] * n_S for _ in range(K+1)]
	if 1 in idx_map:
		dp[0][idx_map[1]] = 1
		
	for k in range(0, K):
		for i in range(n_S):
			if dp[k][i] == 0:
				continue
			v = S[i]
			val = dp[k][i]
			for (u, d) in trans[v]:
				nk = k + 1 + d
				if nk > K:
					continue
				j = idx_map[u]
				dp[nk][j] = (dp[nk][j] + val) % mod
				
	ans = 0
	for k in range(0, K+1):
		for i in range(n_S):
			ans = (ans + dp[k][i]) % mod
			
	print(ans)

if __name__ == '__main__':
	main()