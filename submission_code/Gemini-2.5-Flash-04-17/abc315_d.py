import sys

def solve():
    H, W = map(int, sys.stdin.readline().split())
    cookies = [list(sys.stdin.readline().strip()) for _ in range(H)]

    removed = [[False for _ in range(W)] for _ in range(H)]
    remaining_count = H * W

    # Initially, all rows and columns need to be checked
    rows_to_check = set(range(H))
    cols_to_check = set(range(W))

    while True:
        marked_positions = set()
        newly_removed_indices = set()

        # Step 1: Mark rows
        current_rows_to_check = rows_to_check
        rows_to_check = set() # Reset for next iteration

        for r in current_rows_to_check:
            active_cols = []
            active_colors = set()
            
            # Find active cookies in this row and their colors
            for c in range(W):
                if not removed[r][c]:
                    active_cols.append(c)
                    active_colors.add(cookies[r][c])

            # Check marking condition: >= 2 active cookies and all have the same color
            if len(active_cols) >= 2 and len(active_colors) == 1:
                # Mark all active cookies in this row
                for c in active_cols:
                    marked_positions.add((r, c))

        # Step 2: Mark columns
        current_cols_to_check = cols_to_check
        cols_to_check = set() # Reset for next iteration

        for c in current_cols_to_check:
            active_rows = []
            active_colors = set()

            # Find active cookies in this column and their colors
            for r in range(H):
                if not removed[r][c]:
                    active_rows.append(r)
                    active_colors.add(cookies[r][c])

            # Check marking condition: >= 2 active cookies and all have the same color
            if len(active_rows) >= 2 and len(active_colors) == 1:
                # Mark all active cookies in this column
                for r in active_rows:
                    marked_positions.add((r, c))

        # Step 3: Remove marked cookies
        if not marked_positions:
            break # Terminate if no cookies were marked in this iteration

        for r, c in marked_positions:
            if not removed[r][c]: # Ensure we only remove each cookie once
                removed[r][c] = True
                remaining_count -= 1
                newly_removed_indices.add((r, c))

        # Determine rows/columns to check in the next iteration
        # These are the rows/columns where cookies were just removed, as
        # their status regarding monochromaticity might have changed.
        for r, c in newly_removed_indices:
            rows_to_check.add(r)
            cols_to_check.add(c)

    print(remaining_count)

solve()