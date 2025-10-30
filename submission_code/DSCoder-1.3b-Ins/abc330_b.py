N, L, R = map(int, input().split())
A = list(map(int, input().split()))

X = [0]*N

for i in range(N):
    X[i] = max(L, min(R, A[i]))

print(*X)