def count_sequences(grid, target, memo=None):
    if memo is None:
        memo = {}
    
    grid_tuple = tuple(grid)
    if grid_tuple in memo:
        return memo[grid_tuple]
    
    if grid == target:
        return 1
    
    result = 0
    n = len(grid)
    MOD = 998244353
    
    for l in range(n):
        for r in range(l + 2, n):  # Ensures l+1 < r
            if grid[l] == grid[r]:
                # Check if all values from l+1 to r-1 are different from grid[l]
                if all(grid[i] != grid[l] for i in range(l+1, r)):
                    # Apply the operation
                    new_grid = grid.copy()
                    for i in range(l+1, r):
                        new_grid[i] = grid[l]
                    
                    # Recursively count sequences
                    result = (result + count_sequences(new_grid, target, memo)) % MOD
    
    memo[grid_tuple] = result
    return result

def main():
    n = int(input())
    target = list(map(int, input().split()))
    
    # Initialize the grid with i mod 2 values
    initial_grid = [(i+1) % 2 for i in range(n)]
    
    result = count_sequences(initial_grid, target)
    print(result)

if __name__ == "__main__":
    main()