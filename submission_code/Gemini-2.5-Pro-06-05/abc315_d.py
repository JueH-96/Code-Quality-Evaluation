# YOUR CODE HERE
import sys
from collections import deque

def solve():
    """
    Solves the cookie removal problem by efficiently simulating the process.
    """
    # Fast I/O for large inputs
    input = sys.stdin.readline

    # Read dimensions and grid
    try:
        line = input()
        if not line: return  # Handle empty input
        H, W = map(int, line.split())
        grid = [input().strip() for _ in range(H)]
    except (IOError, ValueError):
        # Handle malformed or empty input
        return

    # --- State Initialization ---
    row_count = [W] * H
    col_count = [H] * W
    total_cookies = H * W

    # Color counts for each row and column. 'a' -> 0, 'b' -> 1, ...
    row_color_counts = [[0] * 26 for _ in range(H)]
    col_color_counts = [[0] * 26 for _ in range(W)]

    for r in range(H):
        for c in range(W):
            color_idx = ord(grid[r][c]) - ord('a')
            row_color_counts[r][color_idx] += 1
            col_color_counts[c][color_idx] += 1

    # Work queues for rows/cols that are monochromatic and have >= 2 cookies
    row_q = deque()
    col_q = deque()

    # Initial population of the work queues
    for r in range(H):
        if row_count[r] >= 2:
            for i in range(26):
                if row_color_counts[r][i] == row_count[r]:
                    row_q.append(r)
                    break

    for c in range(W):
        if col_count[c] >= 2:
            for i in range(26):
                if col_color_counts[c][i] == col_count[c]:
                    col_q.append(c)
                    break

    # Mask to track removed cookies
    removed = [[False] * W for _ in range(H)]

    # --- Simulation Loop ---
    # The loop continues as long as there are rows or columns to process.
    # Each iteration simulates one round of marking and removing.
    while row_q or col_q:
        coords_to_remove = set()

        # Gather cookies from all rows currently in the queue
        while row_q:
            r = row_q.popleft()
            for c in range(W):
                if not removed[r][c]:
                    coords_to_remove.add((r, c))

        # Gather cookies from all columns currently in the queue
        while col_q:
            c = col_q.popleft()
            for r in range(H):
                if not removed[r][c]:
                    coords_to_remove.add((r, c))

        if not coords_to_remove:
            continue

        # Keep track of affected lines to check them for new removals
        affected_rows = set()
        affected_cols = set()

        # Remove the cookies and update the state
        for r, c in coords_to_remove:
            if not removed[r][c]:
                removed[r][c] = True
                total_cookies -= 1
                color_idx = ord(grid[r][c]) - ord('a')

                # Update counts
                row_count[r] -= 1
                col_count[c] -= 1
                row_color_counts[r][color_idx] -= 1
                col_color_counts[c][color_idx] -= 1

                affected_rows.add(r)
                affected_cols.add(c)
        
        # Check affected rows and columns to see if they've become removable
        for r in affected_rows:
            if row_count[r] >= 2:
                # A row is removable if all its remaining cookies are the same color.
                # This is true if the count of one color equals the total count of cookies in the row.
                for i in range(26):
                    if row_color_counts[r][i] == row_count[r]:
                        row_q.append(r)
                        break

        for c in affected_cols:
            if col_count[c] >= 2:
                for i in range(26):
                    if col_color_counts[c][i] == col_count[c]:
                        col_q.append(c)
                        break

    # --- Output ---
    print(total_cookies)

# Call the main logic function
solve()