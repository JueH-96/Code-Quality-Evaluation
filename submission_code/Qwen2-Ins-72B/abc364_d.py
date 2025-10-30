import bisect

N, Q = map(int, input().split())
A = list(map(int, input().split()))
B = [list(map(int, input().split())) for _ in range(Q)]

A.sort()
absA = [(a, i) for i, a in enumerate(A)]
absA.sort(key=lambda x: abs(x[0]))
absA = [a[1] for a in absA]

for b, k in B:
    i = bisect.bisect_left(absA, (N - k))
    d = abs(A[absA[i]] - b)
    print(d)