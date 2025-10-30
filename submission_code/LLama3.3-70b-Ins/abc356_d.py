def popcount(x):
    """Count the number of 1s in the binary representation of x."""
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count

def sum_popcount(N, M):
    """Compute the sum of popcount(k & M) for k from 0 to N."""
    MOD = 998244353
    total = 0
    for k in range(N + 1):
        total = (total + popcount(k & M)) % MOD
    return total

# Read inputs from stdin
N, M = map(int, input().split())

# Compute and print the result
result = sum_popcount(N, M)
print(result)