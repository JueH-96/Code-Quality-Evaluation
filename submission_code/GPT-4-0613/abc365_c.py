import bisect
N, M = map(int, input().split())
A = list(map(int, input().split()))
A.sort()
B = [0]*(N+1)
for i in range(N-1, -1, -1):
    B[i] = B[i+1] + A[i]
ok = 0
ng = 10**9 + 1
while ng - ok > 1:
    mid = (ok + ng) // 2
    idx = bisect.bisect_right(A, mid)
    if idx == N:
        total = mid * N
    else:
        total = mid * idx + B[idx]
    if total <= M:
        ok = mid
    else:
        ng = mid
print(ok)