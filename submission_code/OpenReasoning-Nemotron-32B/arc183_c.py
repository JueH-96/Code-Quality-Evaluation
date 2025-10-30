mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	conditions = []
	for _ in range(m):
		L = int(next(it))
		R = int(next(it))
		X = int(next(it))
		conditions.append((L, R, X))
	
	for (L, R, X) in conditions:
		if L == R and R == X:
			print(0)
			return
			
	conds_by_p = [[] for _ in range(n+1)]
	for (L, R, X) in conditions:
		if L <= X <= R:
			conds_by_p[X].append((L, R))
			
	INF_NEG = -10**9
	maxL_prefix = [None] * (n+1)
	for p in range(1, n+1):
		arr = [INF_NEG] * (n+1)
		for (L, R) in conds_by_p[p]:
			if L > arr[R]:
				arr[R] = L
		max_so_far = INF_NEG
		maxL_arr = [INF_NEG] * (n+1)
		for r in range(1, n+1):
			if arr[r] > max_so_far:
				max_so_far = arr[r]
			maxL_arr[r] = max_so_far
		maxL_prefix[p] = maxL_arr

	dp = [[0] * (n+2) for _ in range(n+2)]
	for i in range(1, n+2):
		for j in range(0, i):
			dp[i][j] = 1
			
	for length in range(1, n+1):
		for l in range(1, n - length + 2):
			r = l + length - 1
			total = 0
			for p in range(l, r+1):
				if maxL_prefix[p][r] < l:
					left_ways = dp[l][p-1]
					right_ways = dp[p+1][r]
					total = (total + left_ways * right_ways) % mod
			dp[l][r] = total % mod

	print(dp[1][n] % mod)

if __name__ == '__main__':
	main()