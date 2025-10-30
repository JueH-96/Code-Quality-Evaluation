import sys

# Read input from stdin
input = sys.stdin.read
data = input().split()

# Extract N and K
N, K = int(data[0]), int(data[1])

# Extract the sequence A
A = list(map(int, data[2:]))

# Filter multiples of K and divide them by K
result = [a // K for a in A if a % K == 0]

# Print the result in ascending order with spaces in between
print(' '.join(map(str, sorted(result))))