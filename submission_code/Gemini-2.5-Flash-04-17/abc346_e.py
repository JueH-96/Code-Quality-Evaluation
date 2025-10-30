import sys
from collections import defaultdict

# Read input
h, w, m = map(int, sys.stdin.readline().split())
ops = []
for _ in range(m):
    t, a, x = map(int, sys.stdin.readline().split())
    ops.append((t, a, x))

# Process operations in reverse
painted_rows = set()
painted_cols = set()
color_counts = defaultdict(int)

# Iterate through operations from last to first
for t, a, x in reversed(ops):
    if t == 1:  # Row operation
        # If this row has already been covered by a later operation (processed earlier in reverse), skip
        if a in painted_rows:
            continue

        # This is the last operation for this row among those not covered by later operations.
        # Cells in this row that are not yet covered by a column operation (processed earlier in reverse)
        # are painted by this operation.
        unpainted_cols_count = w - len(painted_cols)
        if unpainted_cols_count > 0:
            color_counts[x] += unpainted_cols_count

        # Mark this row as covered by a relevant operation
        painted_rows.add(a)

    elif t == 2:  # Column operation
        # If this column has already been covered by a later operation (processed earlier in reverse), skip
        if a in painted_cols:
            continue

        # This is the last operation for this column among those not covered by later operations.
        # Cells in this column that are not yet covered by a row operation (processed earlier in reverse)
        # are painted by this operation.
        unpainted_rows_count = h - len(painted_rows)
        if unpainted_rows_count > 0:
            color_counts[x] += unpainted_rows_count

        # Mark this column as covered by a relevant operation
        painted_cols.add(a)

# Count cells that were never painted by any operation (retain color 0)
# These are cells (r, c) where row r was not covered by a row op AND col c was not covered by a col op
uncovered_rows_count = h - len(painted_rows)
uncovered_cols_count = w - len(painted_cols)
initial_color_cells = uncovered_rows_count * uncovered_cols_count

if initial_color_cells > 0:
    color_counts[0] += initial_color_cells

# Prepare output
# Filter out colors with count 0 and sort by color number
result_colors = sorted([(color, count) for color, count in color_counts.items() if count > 0])

# Print output
print(len(result_colors))
for color, count in result_colors:
    print(color, count)