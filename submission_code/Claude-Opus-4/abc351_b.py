# YOUR CODE HERE
N = int(input())

# Read grid A
grid_A = []
for i in range(N):
    grid_A.append(input().strip())

# Read grid B
grid_B = []
for i in range(N):
    grid_B.append(input().strip())

# Find the differing cell
for i in range(N):
    for j in range(N):
        if grid_A[i][j] != grid_B[i][j]:
            # Output 1-indexed position
            print(i + 1, j + 1)
            break