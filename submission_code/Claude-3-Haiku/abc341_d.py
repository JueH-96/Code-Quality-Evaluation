import math

# Read input
N, M, K = map(int, input().split())

# Find the least common multiple (LCM) of N and M
lcm = (N * M) // math.gcd(N, M)

# Initialize the counter
count = 0

# Iterate through the positive integers
i = 1
while count < K:
    # Check if the current integer is divisible by exactly one of N and M
    if (i % N == 0 and i % M != 0) or (i % N != 0 and i % M == 0):
        count += 1
    i += 1

# Print the answer
print(i - 1)