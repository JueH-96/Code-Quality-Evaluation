from bisect import bisect_left, bisect_right

N, M, D = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort()
B.sort()

ans = -1
for a in A:
    i = bisect_right(B, a + D)
    if i > 0:
        ans = max(ans, a + B[i-1])
    i = bisect_left(B, a - D)
    if i < M:
        ans = max(ans, a + B[i])

print(ans)