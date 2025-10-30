# Read input
H, W = map(int, input().split())
grid = [list(map(int, list(input()))) for _ in range(H)]

# Perform operations
min_sum = float('inf')
for x in range(1, H+1):
    for y in range(1, W+1):
        # Perform operation X
        new_grid = [row[:] for row in grid]
        for i in range(H):
            new_grid[i][y-1] = 1 - new_grid[i][y-1]
        # Perform operation Y
        new_grid2 = [row[:] for row in new_grid]
        for j in range(W):
            new_grid2[x-1][j] = 1 - new_grid2[x-1][j]
        # Calculate the sum
        curr_sum = sum(sum(row) for row in new_grid2)
        min_sum = min(min_sum, curr_sum)

# Print the answer
print(min_sum)