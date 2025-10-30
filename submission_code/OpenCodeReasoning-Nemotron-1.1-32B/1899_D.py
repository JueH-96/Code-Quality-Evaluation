import sys
from collections import Counter

t = int(sys.stdin.readline())
for _ in range(t):
	n = int(sys.stdin.readline())
	arr = list(map(int, sys.stdin.readline().split()))
	cnt = Counter(arr)
	ans = 0
	for v in cnt.values():
		ans += v * (v - 1) // 2
	if 1 in cnt and 2 in cnt:
		ans += cnt[1] * cnt[2]
	print(ans)