# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])

MOD = 998244353

# The expected value of the position of the black ball after K operations
# can be calculated using the formula:
# E(X) = (N + 1) / 2 + (N - 1) * (N + 1) / 4 * (K - 1)
# where E(X) is the expected value, N is the number of balls, and K is the number of operations.

expected_value = (N + 1) * (N + 1) // 4 * (K - 1) % MOD
expected_value = (expected_value + (N + 1) // 2) % MOD

print(expected_value)