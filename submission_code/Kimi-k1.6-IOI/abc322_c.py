import bisect

n, m = map(int, input().split())
A = list(map(int, input().split()))

for i in range(1, n+1):
    idx = bisect.bisect_left(A, i)
    print(A[idx] - i)