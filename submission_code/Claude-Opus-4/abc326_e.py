MOD = 998244353

def modinv(a, m=MOD):
    return pow(a, m - 2, m)

N = int(input())
A = [0] + list(map(int, input().split()))  # 1-indexed

# E[i] = expected salary starting from state x = i
E = [0] * (N + 1)

# Compute E[i] for i from N-1 down to 0
inv_N = modinv(N)
for i in range(N - 1, -1, -1):
    E[i] = (A[i + 1] + (N + 1) * E[i + 1]) * inv_N % MOD

print(E[0])