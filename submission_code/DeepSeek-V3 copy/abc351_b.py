# Read the size of the grid
N = int(input())

# Read grid A
grid_A = [input().strip() for _ in range(N)]

# Read grid B
grid_B = [input().strip() for _ in range(N)]

# Iterate through each cell to find the differing one
for i in range(N):
    for j in range(N):
        if grid_A[i][j] != grid_B[i][j]:
            print(i+1, j+1)
            exit()