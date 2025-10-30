MOD = 998244353

n = int(input())
a = list(map(int, input().split()))

# Compute the number of digits for each element
d = [len(str(x)) for x in a]

# Compute 10^d_j mod MOD for each element
pow10 = [pow(10, di, MOD) for di in d]

# Compute the suffix sum array
suffix = [0] * (n + 1)
for j in range(n-1, -1, -1):
    suffix[j] = (pow10[j] + suffix[j+1]) % MOD

# Calculate S1: sum of a[i] * suffix[i+1] for all i
s1 = 0
for i in range(n):
    s1 = (s1 + a[i] * suffix[i+1]) % MOD

# Calculate S2: sum of a[j] * j for all j
s2 = 0
for j in range(n):
    s2 = (s2 + a[j] * j) % MOD

# Total result
total = (s1 + s2) % MOD
print(total)