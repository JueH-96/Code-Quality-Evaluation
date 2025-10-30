n = int(input().strip())
dist = [[0] * n for _ in range(n)]

for i in range(n-1):
	data = list(map(int, input().split()))
	for idx, j in enumerate(range(i+1, n)):
		dist[i][j] = data[idx]
		dist[j][i] = data[idx]

dp = [-10**18] * (1 << n)
dp[0] = 0

for mask in range(1 << n):
	if dp[mask] == -10**18:
		continue
	for i in range(n):
		if mask & (1 << i):
			continue
		for j in range(i+1, n):
			if mask & (1 << j):
				continue
			new_mask = mask | (1 << i) | (1 << j)
			total = dp[mask] + dist[i][j]
			if total > dp[new_mask]:
				dp[new_mask] = total

print(max(dp))