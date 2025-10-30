N, X = map(int, input().split())
A = list(map(int, input().split()))

A.sort()
X -= sum(A[1:-1])
if X < 0 or X > 100:
    print(-1)
else:
    print(X)