import sys
from collections import defaultdict

def main():
    H, W, M = map(int, sys.stdin.readline().split())
    
    operations = []
    for _ in range(M):
        T, A, X = map(int, sys.stdin.readline().split())
        operations.append((T, A, X))
        
    color_counts = defaultdict(int)
    
    # Set of row indices that have been definitively painted by a row operation
    # (specifically, the latest row operation for that row index, considering reverse iteration).
    used_rows = set()
    # Set of column indices that have been definitively painted by a column operation
    # (specifically, the latest column operation for that column index, considering reverse iteration).
    used_cols = set()
    
    # When processing operations in reverse:
    # `num_available_rows_for_col_op`: Number of rows not yet "claimed" by a (later) row operation.
    # A column operation can paint cells in these rows.
    num_available_rows_for_col_op = H
    # `num_available_cols_for_row_op`: Number of columns not yet "claimed" by a (later) column operation.
    # A row operation can paint cells in these columns.
    num_available_cols_for_row_op = W

    # This counts cells whose color is determined by one of the operations
    # when processed in this reverse loop. It's used to find how many cells remain color 0 by default.
    cells_determined_by_ops = 0

    # Iterate operations in reverse order (from M-th down to 1st)
    for i in range(M - 1, -1, -1):
        T, A, X = operations[i] # A is 1-indexed row/column, X is color
        
        if T == 1: # Row operation
            if A not in used_rows:
                # This operation is the latest for row A that we encounter (iterating backwards).
                # It will paint all cells in row A, *except* for those
                # in columns that have already been "claimed" by a LATER column operation.
                # The number of columns available to be painted by this row op is num_available_cols_for_row_op.
                if num_available_cols_for_row_op > 0: # If all columns are "taken", this row op paints 0 cells.
                    color_counts[X] += num_available_cols_for_row_op
                    cells_determined_by_ops += num_available_cols_for_row_op
                
                used_rows.add(A)
                # This row is now "claimed" by this row operation.
                # It's no longer available for any (even earlier) column operations to paint.
                num_available_rows_for_col_op -= 1
        
        else: # T == 2, Column operation
            if A not in used_cols:
                # This operation is the latest for column A that we encounter.
                # It will paint all cells in column A, *except* for those
                # in rows that have already been "claimed" by a LATER row operation.
                # The number of rows available to be painted by this col op is num_available_rows_for_col_op.
                if num_available_rows_for_col_op > 0: # If all rows are "taken", this col op paints 0 cells.
                    color_counts[X] += num_available_rows_for_col_op
                    cells_determined_by_ops += num_available_rows_for_col_op
                
                used_cols.add(A)
                # This column is now "claimed" by this column operation.
                # It's no longer available for any (even earlier) row operations to paint.
                num_available_cols_for_row_op -= 1
        
    total_cells = H * W
    # Cells not covered by any "winning" operation processed above remain color 0 by default.
    cells_default_zero = total_cells - cells_determined_by_ops
    
    if cells_default_zero > 0:
        color_counts[0] += cells_default_zero # Add to existing count for color 0 if any
        
    output_list = []
    for color_val, count_val in color_counts.items():
        if count_val > 0: # Only include colors that are present on the grid (count > 0)
            output_list.append((color_val, count_val))
            
    output_list.sort() # Sort by color number
    
    sys.stdout.write(str(len(output_list)) + "
")
    for color_val, count_val in output_list:
        sys.stdout.write(f"{color_val} {count_val}
")

if __name__ == '__main__':
    main()