import sys
input = sys.stdin.read

MOD = 998244353

def count_valid_grids(H, W, S):
    # Convert the grid into a 2D list
    grid = [list(row) for row in S]

    # Function to check if the grid is valid
    def is_valid(grid):
        for i in range(H):
            for j in range(W):
                if grid[i][j] == '?':
                    return False
                if j < W - 1 and grid[i][j] == grid[i][j + 1]:
                    return False
                if i < H - 1 and grid[i][j] == grid[i + 1][j]:
                    return False
        return True

    # Function to fill the grid with 1, 2, or 3
    def fill_grid(i, j):
        if i == H:
            return 1 if is_valid(grid) else 0
        if j == W:
            return fill_grid(i + 1, 0)
        if grid[i][j] == '?':
            count = 0
            for num in '123':
                grid[i][j] = num
                count += fill_grid(i, j + 1)
                count %= MOD
            grid[i][j] = '?'
            return count
        else:
            return fill_grid(i, j + 1)

    return fill_grid(0, 0)

# Read input
data = input().split()
H = int(data[0])
W = int(data[1])
S = data[2:]

# Solve the problem
result = count_valid_grids(H, W, S)

# Print the result
print(result)