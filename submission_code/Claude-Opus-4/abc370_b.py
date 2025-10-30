# YOUR CODE HERE
N = int(input())

# Read the triangular matrix
A = [[0] * (N + 1) for _ in range(N + 1)]

for i in range(1, N + 1):
    row = list(map(int, input().split()))
    for j in range(1, i + 1):
        A[i][j] = row[j - 1]

# Start with element 1
current = 1

# Combine with elements 1, 2, ..., N in order
for next_elem in range(1, N + 1):
    if current >= next_elem:
        current = A[current][next_elem]
    else:
        current = A[next_elem][current]

print(current)