import bisect

n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort()
B.sort()

k_candidates = []

for i in range(m):
    b = B[i]
    j = bisect.bisect_left(A, b)
    if j >= n:
        print(-1)
        exit()
    k_candidates.append(j - i)

k = max(k_candidates)

if k + m > n:
    print(-1)
else:
    total = sum(A[k : k + m])
    print(total)