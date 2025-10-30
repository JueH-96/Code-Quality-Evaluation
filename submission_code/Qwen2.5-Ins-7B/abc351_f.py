# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
from collections import defaultdict
cnt = defaultdict(int)
s = 0
ans = 0
for x in a:
    ans += s * x - s + cnt[x] * x - cnt[x]
    s += x
    cnt[x] += 1
print(ans)