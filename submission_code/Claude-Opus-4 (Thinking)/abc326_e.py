MOD = 998244353

def modinv(a, m=MOD):
    return pow(a, m - 2, m)

N = int(input())
A = [0] + list(map(int, input().split()))  # A[1] to A[N]

E = [0] * (N + 1)
N_inv = modinv(N, MOD)

S = 0
for i in range(N - 1, -1, -1):
    S = (S + A[i + 1] + E[i + 1]) % MOD
    E[i] = (S * N_inv) % MOD

print(E[0])