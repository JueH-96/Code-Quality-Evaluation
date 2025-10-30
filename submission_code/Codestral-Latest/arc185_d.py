MOD = 998244353

def expected_operations(N, M):
    total_vertices = N * M + 1
    expected_steps = 0

    for i in range(1, total_vertices):
        # Probability of moving to an unpainted vertex
        prob_move = 1 / 2 if i % N != 0 else 1
        # Expected steps to paint the current vertex
        expected_steps += prob_move * (i + 1)

    return expected_steps % MOD

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

# Calculate and print the result
result = expected_operations(N, M)
print(result)