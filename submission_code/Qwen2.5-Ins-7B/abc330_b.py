# N, L, R = map(int, input().split())
# A = list(map(int, input().split()))

N, L, R = 5, 4, 7
A = [3, 1, 4, 9, 7]

X = [L] * N
for i in range(N):
    if L <= A[i] <= R:
        X[i] = A[i]
    elif A[i] < L:
        X[i] = L
    else:
        X[i] = R

print(*X)