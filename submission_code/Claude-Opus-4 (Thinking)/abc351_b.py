# YOUR CODE HERE
# Read input
n = int(input())
grid_a = []
for _ in range(n):
    grid_a.append(input())

grid_b = []
for _ in range(n):
    grid_b.append(input())

# Find the differing cell
for i in range(n):
    for j in range(n):
        if grid_a[i][j] != grid_b[i][j]:
            # Output 1-indexed position
            print(i + 1, j + 1)
            exit()