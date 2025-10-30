# YOUR CODE HERE

H, W, Q = map(int, input().split())

# Initialize the grid with all walls
grid = [[1]*W for _ in range(H)]

# Process the queries
for _ in range(Q):
    R_q, C_q = map(int, input().split())
    R_q -= 1
    C_q -= 1

    # If there is a wall at (R_q, C_q), destroy that wall
    if grid[R_q][C_q] == 1:
        grid[R_q][C_q] = 0
    else:
        # Destroy the first walls that appear when looking up, down, left, and right from (R_q, C_q)
        for i in range(R_q):
            if grid[i][C_q] == 1:
                grid[i][C_q] = 0
                break
        for i in range(R_q+1, H):
            if grid[i][C_q] == 1:
                grid[i][C_q] = 0
                break
        for j in range(C_q):
            if grid[R_q][j] == 1:
                grid[R_q][j] = 0
                break
        for j in range(C_q+1, W):
            if grid[R_q][j] == 1:
                grid[R_q][j] = 0
                break

# Count the remaining walls
count = sum(sum(row) for row in grid)
print(count)