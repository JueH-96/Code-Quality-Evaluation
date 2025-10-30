MOD = 998244353

n = int(input())
A = list(map(int, input().split()))

pow10 = []
for a in A:
    d = len(str(a))
    pow10.append(pow(10, d, MOD))

# Compute prefix sums mod MOD
prefix_sum = [0] * (n + 1)
for i in range(n):
    prefix_sum[i + 1] = (prefix_sum[i] + A[i]) % MOD

Sum1 = 0
for j in range(1, n):
    term = (pow10[j] * prefix_sum[j]) % MOD
    Sum1 = (Sum1 + term) % MOD

Sum2 = 0
for j in range(n):
    term = (A[j] * j) % MOD
    Sum2 = (Sum2 + term) % MOD

total = (Sum1 + Sum2) % MOD
print(total)