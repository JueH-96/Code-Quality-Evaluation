# YOUR CODE HERE
from collections import defaultdict

n = int(input())
a = list(map(int, input().split()))
cnt = defaultdict(int)
total = 0
for i in range(n):
    for j in range(i + 1, n):
        max_val = max(a[i], a[j])
        min_val = min(a[i], a[j])
        ratio = max_val // min_val + (max_val % min_val > 0)
        total += ratio
        cnt[ratio] += 1

for k, v in cnt.items():
    if k > 1:
        total -= v * (v - 1) // 2

print(total)