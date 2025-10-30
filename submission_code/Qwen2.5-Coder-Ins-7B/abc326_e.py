# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

MOD = 998244353

# Initialize the expected value array
E = [0] * (N + 1)

# Calculate the expected value for each state
for i in range(N, 0, -1):
    E[i] = (A[i - 1] + sum(E[j] for j in range(i + 1, N + 1))) % MOD

# The expected value for the initial state (x=0) is E[1]
result = E[1]

# Print the result
print(result)