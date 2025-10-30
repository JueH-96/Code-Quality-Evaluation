import sys

def solve():
    # Read input
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]

    # Find the minimal bounding box (0-indexed) of all '#' cells
    # Initialize with values outside the possible range [0, H-1] and [0, W-1]
    min_r, max_r = H, -1
    min_c, max_c = W, -1

    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                min_r = min(min_r, i)
                max_r = max(max_r, i)
                min_c = min(min_c, j)
                max_c = max(max_c, j)

    # The problem guarantees at least one '#', so min_r, max_r, min_c, max_c
    # will be valid indices representing a non-empty bounding box.
    # The candidate rectangle for the final set of black cells is the one
    # defined by these bounds: rows [min_r, max_r] and columns [min_c, max_c] (0-indexed).

    # Check if any '.' cell is found inside this candidate black rectangle.
    # According to the definition of a black rectangle (a,b,c,d), all cells
    # within the rectangle must be black, and all cells outside must be white.
    # If an existing '.' cell is found inside our candidate rectangle, it contradicts
    # the requirement that cells inside the rectangle must be black. Thus, it's impossible.
    # We only need to check the cells within the bounding box [min_r, max_r] x [min_c, max_c].
    # Any '.' outside the bounding box is fine.
    # The ranges for the loops should be inclusive, hence +1.
    # If max_r < min_r (no '#' found), range(min_r, max_r + 1) will be empty, which is correct
    # behavior, although the problem constraints guarantee max_r >= min_r.
    for i in range(min_r, max_r + 1):
        for j in range(min_c, max_c + 1):
            if grid[i][j] == '.':
                 # If a '.' is found within the bounding box, it's impossible.
                print("No")
                return # Exit the function immediately

    # If the loops complete without finding a '.' inside the bounding box,
    # it means the candidate rectangle [min_r, max_r] x [min_c, max_c] is possible.
    # We can achieve this by painting all '?' cells within this box as '#' and
    # all '?' cells outside this box as '.'. This results in a grid where
    # exactly the cells in the [min_r, max_r] x [min_c, max_c] region are a mix
    # of initial '#' and painted '?' (all becoming '#'), and all cells outside
    # are a mix of initial '.' and painted '?' (all becoming '.').
    # This forms a valid black rectangle.
    print("Yes")

solve()