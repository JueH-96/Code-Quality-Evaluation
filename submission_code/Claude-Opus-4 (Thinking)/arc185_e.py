from math import gcd

N = int(input())
A = list(map(int, input().split()))

MOD = 998244353

# Precompute powers of 2
pow2 = [1]
for i in range(N):
    pow2.append(pow2[-1] * 2 % MOD)

# Compute answer for each m
answer = 0
print(0)  # m = 1

for m in range(2, N + 1):
    # Compute S[m] = sum_{1 <= i < m} gcd(A_i, A_m) * 2^(i-1)
    S_m = 0
    for i in range(1, m):
        S_m = (S_m + gcd(A[i-1], A[m-1]) * pow2[i-1]) % MOD
    answer = (2 * answer + S_m) % MOD
    print(answer)