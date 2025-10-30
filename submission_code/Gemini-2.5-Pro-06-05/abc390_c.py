import sys

def solve():
    """
    Reads the grid state from stdin, determines if all black cells can form a
    rectangle by painting the '?' cells, and prints "Yes" or "No" to stdout.
    """
    # Read grid dimensions. Using try-except for robustness against empty input.
    try:
        H, W = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        return
    
    # Read the grid from standard input.
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # Step 1: Find the minimal bounding box of all '#' cells.
    # The problem guarantees at least one '#' exists.
    min_r, max_r = H, -1
    min_c, max_c = W, -1

    for r in range(H):
        for c in range(W):
            if S[r][c] == '#':
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)

    # Step 2: Check if this bounding box can form a valid black rectangle.
    # Any valid black rectangle must contain all '#'s, so it must also contain
    # their minimal bounding box. If this minimal box contains a cell that is
    # already white ('.'), then no solution is possible.
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if S[r][c] == '.':
                # A white cell inside the required rectangle makes it impossible.
                print("No")
                return

    # If no contradictions are found, a valid configuration exists.
    # We can paint all '?' inside the box black and all '?' outside white.
    print("Yes")

# Execute the solution
solve()