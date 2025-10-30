MOD = 998244353

n = int(input())
A = list(map(int, input().split()))

# Precompute powers of 10 up to 10^10 mod MOD
pow10 = [1] * 11
for i in range(1, 11):
    pow10[i] = (pow10[i-1] * 10) % MOD

# Compute 10^L for each element's length
pow10_L = []
for a in A:
    l = len(str(a))
    pow10_L.append(pow10[l])

# Compute prefix sums
prefix = [0] * (n + 1)
for i in range(n):
    prefix[i+1] = (prefix[i] + A[i]) % MOD  # Mod here to prevent overflow

# Calculate sum1
sum1 = 0
for j in range(n):
    sum1 = (sum1 + prefix[j] * pow10_L[j]) % MOD

# Calculate sum2
sum2 = 0
for j in range(n):
    sum2 = (sum2 + A[j] * j) % MOD

# Compute the final answer
total = (sum1 + sum2) % MOD
print(total)