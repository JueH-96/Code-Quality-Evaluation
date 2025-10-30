from itertools import accumulate
from bisect import bisect_right
N, M = map(int, input().split())
A = list(map(int, input().split()))
S = [0] + list(accumulate(A))
S2 = [0] + list(accumulate(A[::-1]))
ans = 0
for i in range(N):
    x = S[i]
    y = S2[N-i-1]
    if x % M == 0:
        ans += N-i-1
    if y % M == 0:
        ans += i
    if x % M == y % M:
        ans -= 1
    z = bisect_right(S, x+y)
    if S[z-1] % M == 0:
        ans += N-z
    z = bisect_right(S2, x+y)
    if S2[z-1] % M == 0:
        ans += z-1
print(ans//2)