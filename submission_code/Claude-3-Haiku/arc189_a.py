import sys

def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # Initialize the grid with the initial values
    grid = [i % 2 for i in range(1, N+1)]

    # Count the number of valid sequences of operations
    count = 0

    # Iterate through all possible pairs of cells (l, r)
    for l in range(N):
        for r in range(l+2, N):
            # Check if the conditions are satisfied
            if grid[l] == grid[r] and all(grid[i] != grid[l] for i in range(l+1, r)):
                # Replace the values in cells l+1 to r-1 with the value in cell l
                new_grid = grid[:l+1] + [grid[l]] * (r-l-1) + grid[r:]
                
                # Check if the new grid matches the target grid
                if new_grid == A:
                    count += 1

    print(count % 998244353)

if __name__ == "__main__":
    solve()