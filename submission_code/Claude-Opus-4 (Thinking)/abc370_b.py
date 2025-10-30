# YOUR CODE HERE
n = int(input())

# Read the matrix A
A = []
for i in range(n):
    row = list(map(int, input().split()))
    A.append(row)

# Start with element 1
current = 1

# Combine with elements 1, 2, ..., n in order
for j in range(1, n+1):
    if current >= j:
        # Use A[current][j], but convert to 0-indexed
        current = A[current-1][j-1]
    else:
        # Use A[j][current], but convert to 0-indexed
        current = A[j-1][current-1]

print(current)