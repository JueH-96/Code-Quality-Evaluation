# YOUR CODE HERE
from collections import *

N = int(input())
A = list(map(int, input().split()))
cnt = Counter()
ans = 0

for a in A:
    ans += cnt[a] * (cnt[a] - 1) // 2
    cnt[a] += 1
    cnt[A.pop(0)] -= 1

print(ans)