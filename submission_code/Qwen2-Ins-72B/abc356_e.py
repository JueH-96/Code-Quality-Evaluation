from bisect import bisect_left, bisect_right
from math import floor

N = int(input())
A = list(map(int, input().split()))

As = sorted(set(A))
d = {a: i for i, a in enumerate(As)}
A = [d[a] for a in A]

cnt = [0] * N
acc = [0] * N
for i, a in enumerate(A):
    l = bisect_left(A, a, 0, i)
    r = bisect_right(A, a, i + 1, N)
    cnt[a] += i - l
    cnt[a] += r - (i + 1)
    acc[a] += (i - l) * (i - l + 1) // 2
    acc[a] += (r - (i + 1)) * (r - i) // 2

ans = 0
for i, a in enumerate(As):
    if i == 0:
        ans += cnt[i] * a
    elif i == len(As) - 1:
        ans += cnt[i] * (a - 1)
    else:
        ans += (acc[i] - acc[i - 1]) * floor(a / As[i - 1])
        ans += (cnt[i] * (a - 1) - acc[i] + acc[i - 1]) * floor(As[i - 1] / a)
        ans += (acc[i] - acc[i - 1]) * floor(a / As[i - 1])
print(ans)