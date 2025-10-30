def find_difference(N, grid_A, grid_B):
    for i in range(N):
        for j in range(N):
            if grid_A[i][j] != grid_B[i][j]:
                return (i + 1, j + 1)
    return None

# Read input
N = int(input().strip())
grid_A = [input().strip() for _ in range(N)]
grid_B = [input().strip() for _ in range(N)]

# Find the different cell
different_cell = find_difference(N, grid_A, grid_B)

# Print the result
print(f"{different_cell[0]} {different_cell[1]}")