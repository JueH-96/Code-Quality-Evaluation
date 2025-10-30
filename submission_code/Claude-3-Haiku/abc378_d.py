from collections import deque

def count_paths(H, W, K, grid):
    # Create a 3D array to store the number of paths that reach each cell in K steps
    dp = [[[0] * (K+1) for _ in range(W)] for _ in range(H)]

    # Initialize the starting cell (1, 1) with 1 path of 0 steps
    dp[0][0][0] = 1

    # Iterate through the number of steps
    for step in range(1, K+1):
        # Iterate through the grid
        for i in range(H):
            for j in range(W):
                # If the current cell is empty
                if grid[i][j] == '.':
                    # Check the neighboring cells
                    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        # If the neighboring cell is within the grid and is empty
                        if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
                            # Add the number of paths that reach the neighboring cell in step-1 steps
                            dp[i][j][step] += dp[ni][nj][step-1]

    # Return the total number of paths that reach any cell in K steps
    return sum(sum(dp[i][j][K] for i in range(H)) for j in range(W))

# Read the input
H, W, K = map(int, input().split())
grid = [input() for _ in range(H)]

# Solve the problem and print the answer
print(count_paths(H, W, K, grid))