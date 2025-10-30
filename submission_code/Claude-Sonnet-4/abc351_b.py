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

# Find the difference
for i in range(N):
    for j in range(N):
        if grid_A[i][j] != grid_B[i][j]:
            print(i + 1, j + 1)  # Convert to 1-indexed
            break
    else:
        continue
    break