n = int(input().strip())
heights = list(map(int, input().split()))
from collections import defaultdict
groups = defaultdict(list)
for i, h in enumerate(heights):
	groups[h].append(i)

ans = 1
for h, arr in groups.items():
	m = len(arr)
	if m < 2:
		continue
	dp = [dict() for _ in range(m)]
	for i in range(1, m):
		for j in range(i):
			d = arr[i] - arr[j]
			if d in dp[j]:
				dp[i][d] = dp[j][d] + 1
			else:
				dp[i][d] = 2
			if dp[i][d] > ans:
				ans = dp[i][d]

print(ans)