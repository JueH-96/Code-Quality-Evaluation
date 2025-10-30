MOD = 998244353

def mod_inverse(a):
    return pow(a, MOD - 2, MOD)

N = int(input())
A = list(map(int, input().split()))

T = [0] * (N + 2)

for i in range(N, 0, -1):
    T[i] = (A[i - 1] + T[i + 1] * (N + 1) * mod_inverse(N)) % MOD

result = (T[1] * mod_inverse(N)) % MOD
print(result)