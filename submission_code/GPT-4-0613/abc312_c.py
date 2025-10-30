import bisect

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort()
B.sort()

l = 0
r = 10**9 + 1

while r - l > 1:
    X = (l + r) // 2
    if bisect.bisect_left(A, X) >= bisect.bisect_right(B, X):
        r = X
    else:
        l = X

print(r)