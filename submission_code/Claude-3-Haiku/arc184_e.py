import sys

def f(A_i, A_j):
    M = len(A_i)
    x = 0
    while True:
        if A_i == A_j:
            return x
        for k in range(M):
            A_i[k] = (sum(A_i[:k+1])) % 2
            A_j[k] = (sum(A_j[:k+1])) % 2
        x += 1

N, M = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(N)]

total = 0
for i in range(N):
    for j in range(i, N):
        total += f(A[i][:], A[j][:])

print(total % 998244353)