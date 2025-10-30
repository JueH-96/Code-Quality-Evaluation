# YOUR CODE HERE
MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    grid = []
    for i in range(H):
        grid.append(list(data[2 + i]))
    
    # Precompute the positions of '?'
    q_positions = []
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '?':
                q_positions.append((i, j))
    
    # Precompute the fixed cells
    fixed = {}
    for i in range(H):
        for j in range(W):
            if grid[i][j] != '?':
                fixed[(i, j)] = int(grid[i][j])
    
    # Check if the fixed cells already violate the condition
    for (i, j), val in fixed.items():
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x, y = i + dx, j + dy
            if 0 <= x < H and 0 <= y < W:
                if (x, y) in fixed and fixed[(x, y)] == val:
                    print(0)
                    return
    
    # Now, we need to count the number of ways to assign 1, 2, 3 to the '?' positions
    # such that no two adjacent cells have the same value.
    # We can use dynamic programming to count the valid assignments.
    
    # Since the grid is small (H*W <= 200), we can use a bitmask to represent the assignments.
    # However, a more efficient approach is to use a backtracking with memoization.
    
    # We will represent the grid as a list of lists, and for each '?', we will try all possible values.
    # We will use memoization to avoid redundant calculations.
    
    from functools import lru_cache
    
    # Convert the grid to a tuple of tuples for hashing
    grid_tuple = tuple(tuple(row) for row in grid)
    
    @lru_cache(maxsize=None)
    def dp(i, j, prev_row, current_row):
        if i == H:
            return 1
        if j == W:
            return dp(i+1, 0, current_row, tuple())
        
        if grid[i][j] != '?':
            val = int(grid[i][j])
            # Check if the value is valid with the left and top neighbors
            if j > 0 and current_row[j-1] == val:
                return 0
            if i > 0 and prev_row[j] == val:
                return 0
            return dp(i, j+1, prev_row, current_row + (val,))
        else:
            total = 0
            for val in [1, 2, 3]:
                # Check if the value is valid with the left and top neighbors
                if j > 0 and current_row[j-1] == val:
                    continue
                if i > 0 and prev_row[j] == val:
                    continue
                total += dp(i, j+1, prev_row, current_row + (val,))
            return total % MOD
    
    result = dp(0, 0, tuple(), tuple())
    print(result)

if __name__ == "__main__":
    main()