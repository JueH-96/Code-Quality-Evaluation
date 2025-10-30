MOD = 998244353

def expected_position(N, K):
    # The expected position of the black ball after K operations
    # can be calculated using the formula:
    # E(X) = 1 + (N - 1) * (1 - 1/N)^K
    # This is derived from the fact that each swap operation
    # moves the black ball to a new position with equal probability.

    # Calculate (1 - 1/N)^K
    prob = pow(N - 1, K, MOD) * pow(N, MOD - 2, MOD) % MOD

    # Calculate the expected position
    expected = (1 + (N - 1) * prob) % MOD

    return expected

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])

# Calculate and print the expected position
result = expected_position(N, K)
print(result)