import sys
from math import ceil

N = int(input())
H = list(map(int, input().split()))

ans = 0
for h in H:
    if h % 3 == 0:
        ans += 2 * ceil(h / 3) - 1
    else:
        ans += 2 * ceil(h / 3)
print(ans)