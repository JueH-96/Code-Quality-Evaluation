import sys

# Read input
line = sys.stdin.readline().strip().split()
N, K = map(int, line)
A = list(map(int, sys.stdin.readline().strip().split()))

# Compute the sum
MOD = 998244353
total = 0
for l in range(1, N+1):
    for r in range(l, N+1):
        s = sum(A[l-1:r])
        total = (total + pow(s, K, MOD)) % MOD

# Print the answer
print(total)