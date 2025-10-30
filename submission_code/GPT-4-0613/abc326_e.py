MOD = 998244353
N = int(input())
A = list(map(int, input().split()))
A = [0] + A
F = [0]*(N+1)
G = [0]*(N+1)
for i in range(1, N+1):
    F[i] = (F[i-1] + pow(i, MOD-2, MOD)) % MOD
for i in range(N, 0, -1):
    G[i] = (G[i+1] + F[i+1]) % MOD
    A[i] = A[i]*G[i] % MOD
for i in range(N, 0, -1):
    A[i] = (A[i] + A[i+1]) % MOD
print(A[1])