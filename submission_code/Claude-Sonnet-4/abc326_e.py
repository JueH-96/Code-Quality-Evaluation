def modinv(a, m=998244353):
    return pow(a, m-2, m)

N = int(input())
A = list(map(int, input().split()))

MOD = 998244353

# E[i] = expected salary starting from state i
E = [0] * (N + 1)

# E[N] = 0 (terminal state)
E[N] = 0

# Work backwards
for i in range(N-1, -1, -1):
    # E[i] = (1/N) * sum(A[j-1] + E[j] for j in range(i+1, N+1))
    total = 0
    for j in range(i+1, N+1):
        total = (total + A[j-1] + E[j]) % MOD
    
    E[i] = (total * modinv(N)) % MOD

print(E[0])