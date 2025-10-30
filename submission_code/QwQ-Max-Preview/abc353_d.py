MOD = 998244353

N = int(input())
A = list(map(int, input().split()))

# Precompute 10^d mod MOD for d from 1 to 10
pow10 = [1] * 11  # pow10[0] is unused
for d in range(1, 11):
    pow10[d] = (pow10[d-1] * 10) % MOD

# Compute prefix sums
prefix_sum = [0] * N
current_sum = 0
for i in range(N):
    prefix_sum[i] = current_sum % MOD
    current_sum += A[i]
    current_sum %= MOD

first_part = 0
second_part = 0

for j in range(1, N):
    a_j = A[j]
    d = len(str(a_j))
    p = pow10[d]
    term1 = (prefix_sum[j] * p) % MOD
    first_part = (first_part + term1) % MOD
    
    term2 = (j * a_j) % MOD
    second_part = (second_part + term2) % MOD

total = (first_part + second_part) % MOD
print(total)