# YOUR CODE HERE
MOD = 998244353

N, M = map(int, input().split())

# Calculate (M-1)^N mod MOD
result = pow(M - 1, N, MOD)

# Add or subtract (M-1) based on whether N is even or odd
if N % 2 == 0:
    result = (result + (M - 1)) % MOD
else:
    result = (result - (M - 1)) % MOD

print(result)