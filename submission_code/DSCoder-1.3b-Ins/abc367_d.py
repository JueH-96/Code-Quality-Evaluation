# YOUR CODE HERE

# Read the inputs
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Initialize the count of pairs
count = 0

# Iterate over the rest areas
for i in range(N):
    # Calculate the minimum steps to reach the next rest area
    steps = (A[i] + A[i+1]) % M
    # If the steps are not a multiple of M, increment the count
    if steps % M != 0:
        count += 1

# Print the count of pairs
print(count)