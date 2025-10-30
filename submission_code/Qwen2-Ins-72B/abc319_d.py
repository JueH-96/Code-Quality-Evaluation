import sys
from bisect import bisect_left

readline = sys.stdin.readline
write = sys.stdout.write

N, M = map(int, readline().split())
L = list(map(int, readline().split()))

def check(W):
    cnt = 1
    cur = 0
    for i in range(N):
        if cur + L[i] + 1 <= W:
            cur += L[i] + 1
        else:
            cnt += 1
            cur = L[i] + 1
    return cnt <= M

l = max(L)
r = sum(L) + N - 1

while l < r:
    mid = (l + r) // 2
    if check(mid):
        r = mid
    else:
        l = mid + 1

write(str(l))