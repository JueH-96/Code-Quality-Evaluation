# Read the input
N, M = map(int, input().split())

# Initialize the sum
total = 0

# Compute the sum
for k in range(N + 1):
    total += bin(k & M).count('1')
    total %= 998244353

# Print the answer
print(total)