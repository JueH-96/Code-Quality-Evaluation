import sys

# Read all input and split into a list of integers
data = list(map(int, sys.stdin.read().split()))

# Initialize index to 0
index = 0

# Read N from the first element
N = data[index]
index += 1

# Create a 2D list to store the A matrix
A = []

# Read each row of the A matrix
for i in range(1, N + 1):  # i is the row number from 1 to N
    row = []
    for j in range(i):  # Each row i has i elements
        val = data[index]
        row.append(val)
        index += 1
    A.append(row)

# Start with current element as 1
current = 1

# Combine current element with k from 1 to N in order
for k in range(1, N + 1):
    max_ab = max(current, k)
    min_ab = min(current, k)
    # Get the result from A[max_ab-1][min_ab-1]
    result = A[max_ab - 1][min_ab - 1]
    # Update current to the result
    current = result

# Print the final element
print(current)