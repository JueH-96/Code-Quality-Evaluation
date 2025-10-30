from bisect import bisect_right

N, M = map(int, input().split())
A = list(map(int, input().split()))

A.sort()
acc = [0] * (N + 1)
for i in range(N):
    acc[i + 1] = acc[i] + 1

ans = 0
for i in range(N):
    right = bisect_right(A, A[i] + M)
    ans = max(ans, acc[right] - acc[i])
print(ans)