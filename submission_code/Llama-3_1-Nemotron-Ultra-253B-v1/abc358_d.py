import bisect

n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort()
B.sort()

j = 0
total = 0

for b in B:
    k = bisect.bisect_left(A, b, j)
    if k >= len(A):
        print(-1)
        exit()
    total += A[k]
    j = k + 1

print(total)