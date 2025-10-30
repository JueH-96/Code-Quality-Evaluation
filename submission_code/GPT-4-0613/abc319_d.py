import sys
from itertools import accumulate

def check(mid):
    cnt = 1
    now = 0
    for i in range(n):
        if now + l[i] <= mid:
            now += l[i]
        else:
            cnt += 1
            now = l[i]
    return cnt <= m

n, m = map(int, sys.stdin.readline().split())
l = list(map(int, sys.stdin.readline().split()))
l = list(accumulate([0] + l))
l = [l[i+1] - l[i] for i in range(n)]

left = max(l)
right = sum(l) + n
while right - left > 1:
    mid = (left + right) // 2
    if check(mid):
        right = mid
    else:
        left = mid

print(right)