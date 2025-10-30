import sys

# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Sort the lost socks in ascending order
A.sort()

# Initialize the total weirdness to 0
total_weirdness = 0

# Create pairs of socks
for i in range(K):
    if i > 0:
        total_weirdness += A[i] - A[i-1] - 1
    if 2*N - K - 2*i >= 1:
        total_weirdness += A[i] - i - 1

# Print the minimum total weirdness
print(total_weirdness)