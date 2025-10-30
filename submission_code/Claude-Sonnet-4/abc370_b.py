# YOUR CODE HERE
n = int(input())
A = []

# Read the matrix A
for i in range(n):
    row = list(map(int, input().split()))
    A.append(row)

# Start with element 1
current = 1

# Combine with elements 1, 2, ..., N in order
for j in range(1, n + 1):
    # Combine current with j
    if current >= j:
        current = A[current - 1][j - 1]  # Convert to 0-indexed
    else:
        current = A[j - 1][current - 1]  # Convert to 0-indexed

print(current)