import sys
from collections import defaultdict

def solve():
    H, W, M = map(int, sys.stdin.readline().split())

    operations = []
    for _ in range(M):
        t, a, x = map(int, sys.stdin.readline().split())
        operations.append((t, a, x))

    final_counts = defaultdict(int)
    
    # Store row/column indices that have been covered by the *latest* relevant operation
    # (because we process operations in reverse chronological order)
    rows_covered = set()
    cols_covered = set()
    
    # Track the number of distinct rows/columns that have been covered by their latest operation
    num_rows_covered = 0
    num_cols_covered = 0

    # Iterate operations in reverse chronological order
    # This way, the first time we encounter an operation for a specific row/column,
    # it is the "latest" operation for that row/column in the original chronological sequence.
    for i in range(M - 1, -1, -1):
        t, a, x = operations[i]
        
        # Optimization: If all rows are covered OR all columns are covered,
        # then the entire grid's final colors are determined. No further operations
        # can change the color of any cell that hasn't already been accounted for.
        # This is because any new row/column operation would either:
        # 1. Target a row/column already "finalized" by a later operation (and thus skipped by `if a not in ...`)
        # 2. Target a row/column not yet "finalized", but then the number of cells it can affect
        #    is `W - num_cols_covered` or `H - num_rows_covered`. If all columns are covered, `W - W = 0`.
        #    If all rows are covered, `H - H = 0`. So it adds 0 cells.
        # This break prevents unnecessary iterations.
        if num_rows_covered == H or num_cols_covered == W:
            break

        if t == 1: # Row operation (repaint A_i-th row with color X_i)
            # If this row has not been covered by a *later* operation (in original time),
            # then this operation is the one that determines its color (unless overridden by a column op).
            if a not in rows_covered:
                rows_covered.add(a)
                num_rows_covered += 1
                
                # The cells in this row that are not covered by any *later* column operation
                # will take on color X_i.
                # num_cols_covered tracks how many columns HAVE been covered by *later* operations.
                cells_to_paint = W - num_cols_covered
                final_counts[x] += cells_to_paint
                
        else: # Column operation (t == 2) (repaint A_i-th column with color X_i)
            # Similar logic for columns.
            if a not in cols_covered:
                cols_covered.add(a)
                num_cols_covered += 1
                
                # The cells in this column that are not covered by any *later* row operation
                # will take on color X_i.
                # num_rows_covered tracks how many rows HAVE been covered by *later* operations.
                cells_to_paint = H - num_rows_covered
                final_counts[x] += cells_to_paint
    
    # After processing all relevant operations, some cells might still be original color 0.
    # These are cells in rows that were never touched by a row operation,
    # AND in columns that were never touched by a column operation.
    # Equivalently, cells whose row/column was not in `rows_covered`/`cols_covered`.
    uncovered_cells = (H - num_rows_covered) * (W - num_cols_covered)
    if uncovered_cells > 0:
        final_counts[0] += uncovered_cells

    # Prepare output: filter colors with positive counts and sort by color number
    result_list = []
    for color, count in final_counts.items():
        if count > 0:
            result_list.append((color, count))
    
    result_list.sort() # Sorts by color number (the first element of the tuple)

    print(len(result_list))
    for color, count in result_list:
        print(color, count)

solve()