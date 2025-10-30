import sys
from collections import Counter

def solve():
    H, W = map(int, sys.stdin.readline().split())
    
    # initial_grid_colors stores the input character grid
    initial_grid_colors = []
    for _ in range(H):
        initial_grid_colors.append(list(sys.stdin.readline().strip()))

    # grid will store the actual state: color character or None if cookie is removed
    grid = [[initial_grid_colors[i][j] for j in range(W)] for i in range(H)]

    remaining_cookies_count = H * W

    # row_counts[i]: number of remaining cookies in row i
    # col_counts[j]: number of remaining cookies in column j
    row_counts = [W] * H
    col_counts = [H] * W
    
    # row_color_freq[i]: Counter of colors for remaining cookies in row i
    # col_color_freq[j]: Counter of colors for remaining cookies in column j
    row_color_freq = [Counter() for _ in range(H)]
    col_color_freq = [Counter() for _ in range(W)]

    # row_active_indices[i]: set of j-coordinates for remaining cookies in row i
    # col_active_indices[j]: set of i-coordinates for remaining cookies in column j
    row_active_indices = [set(range(W)) for _ in range(H)]
    col_active_indices = [set(range(H)) for _ in range(W)]

    # Populate initial frequency counters and active indices
    for i in range(H):
        for j in range(W):
            color = initial_grid_colors[i][j]
            row_color_freq[i][color] += 1
            col_color_freq[j][color] += 1

    # Queues for rows/columns that need to be re-checked in the current iteration.
    # Initially, all rows and columns might qualify.
    rows_to_check = set(range(H))
    cols_to_check = set(range(W))

    # Main simulation loop
    while rows_to_check or cols_to_check:
        # Get the rows/columns to process in this round and clear queues for next round
        current_rows_to_check = rows_to_check.copy()
        current_cols_to_check = cols_to_check.copy()
        rows_to_check.clear()
        cols_to_check.clear()
        
        # Set to collect coordinates of cookies to be removed in this round
        to_remove_coords = set()

        # Step 1: Mark cookies based on row conditions
        for r in current_rows_to_check:
            # A row needs to be marked if:
            # 1. It has 2 or more cookies remaining (`row_counts[r] >= 2`).
            # 2. All remaining cookies have the same color (`len(row_color_freq[r]) == 1`).
            if row_counts[r] >= 2 and len(row_color_freq[r]) == 1:
                # If the row qualifies, all its remaining cookies are marked.
                for c in row_active_indices[r]:
                    to_remove_coords.add((r, c))

        # Step 2: Mark cookies based on column conditions
        for c in current_cols_to_check:
            # A column needs to be marked if:
            # 1. It has 2 or more cookies remaining (`col_counts[c] >= 2`).
            # 2. All remaining cookies have the same color (`len(col_color_freq[c]) == 1`).
            if col_counts[c] >= 2 and len(col_color_freq[c]) == 1:
                # If the column qualifies, all its remaining cookies are marked.
                for r in col_active_indices[c]:
                    to_remove_coords.add((r, c))
        
        # Step 3: Check if any cookies were marked. If not, terminate.
        if not to_remove_coords:
            break

        # Step 3 (cont.): Remove marked cookies and update state
        for r, c in to_remove_coords:
            # A cookie might be marked by both a row and a column in the same round.
            # Ensure it's only processed/removed if it hasn't been already.
            if grid[r][c] is not None:
                color = grid[r][c] # Store color before setting to None
                grid[r][c] = None # Mark as removed
                remaining_cookies_count -= 1

                # Update counts and frequencies for the affected row and column
                row_counts[r] -= 1
                col_counts[c] -= 1
                
                row_color_freq[r][color] -= 1
                if row_color_freq[r][color] == 0: # If count drops to 0, remove color entry
                    del row_color_freq[r][color]
                
                col_color_freq[c][color] -= 1
                if col_color_freq[c][color] == 0: # If count drops to 0, remove color entry
                    del col_color_freq[c][color]
                
                # Remove from active indices sets
                row_active_indices[r].remove(c)
                col_active_indices[c].remove(r)

                # Add affected row and column to next check queues.
                # Their state has changed, so they might now qualify or affect other checks.
                rows_to_check.add(r)
                cols_to_check.add(c)
                
    # Print the final count of remaining cookies
    sys.stdout.write(str(remaining_cookies_count) + "
")

solve()