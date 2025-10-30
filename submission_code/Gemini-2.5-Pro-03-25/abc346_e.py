# YOUR CODE HERE
import sys

# Ensure that reading large inputs is efficient
# Assign sys.stdin.readline to input for faster reading
input = sys.stdin.readline

def solve():
    # Read grid dimensions H, W and number of operations M from input
    H, W, M = map(int, input().split())
    
    # Read all M operations and store them
    # Each operation is a tuple (T_i, A_i, X_i)
    operations = []
    for _ in range(M):
        # Store as tuple
        operations.append(tuple(map(int, input().split())))

    # Dictionary to store counts of each color present in the final grid
    # Keys are color numbers, values are counts of cells with that color.
    color_counts = {}
    
    # Sets to keep track of rows and columns whose final color has been determined
    # by processing operations in reverse. If a row/column index is in the set,
    # it means a later operation (in the original sequence) has finalized its state.
    used_rows = set()
    used_cols = set()
    
    # Variable to accumulate the total count of cells painted with non-zero colors.
    # This helps calculate the final count for color 0.
    total_painted_non_zero = 0 

    # Iterate through operations in reverse order (from M-1 down to 0 index, corresponding to operation M down to 1)
    # This allows determining the final color of cells efficiently.
    # The last operation affecting a row/column determines its final state for cells
    # not affected by an even later operation on the other dimension.
    for i in range(M - 1, -1, -1):
        T, A, X = operations[i]
        
        if T == 1: # Row operation: Paint row A with color X
            # Check if this row's final state has already been determined by a later operation
            # If A is already in used_rows, it means an operation with index > i painted row A.
            # That later operation takes precedence, so we skip this operation.
            if A not in used_rows:
                # Mark this row as processed (final state determined by this or later operation)
                used_rows.add(A)
                
                # Calculate the number of cells in this row that are *not* part of columns
                # whose final state has already been determined by later column operations.
                # These are the cells potentially affected by *this* row operation.
                num_cells = W - len(used_cols)
                
                # If there are any such cells (i.e., not all columns are already finalized)
                if num_cells > 0:
                    # If the color X is non-zero
                    if X != 0: 
                       # Update the count for this color X
                       # Use .get(X, 0) to handle the first time a color appears
                       color_counts[X] = color_counts.get(X, 0) + num_cells
                       # Add the number of affected cells to the total count of non-zero painted cells
                       total_painted_non_zero += num_cells
                    # If X == 0, these cells will finally have color 0.
                    # We don't update color_counts for color 0 here.
                    # Instead, the final count for color 0 is calculated based on total cells
                    # minus the cells painted non-zero. So, no action needed for X=0 here.
        
        else: # T == 2, Column operation: Paint column A with color X
            # Check if this column's final state has already been determined by a later operation
            if A not in used_cols:
                # Mark this column as processed
                used_cols.add(A)
                
                # Calculate the number of cells in this column that are *not* part of rows
                # whose final state has already been determined by later row operations.
                num_cells = H - len(used_rows)

                # If there are any such cells
                if num_cells > 0:
                   # If the color X is non-zero
                   if X != 0:
                      # Update the count for this color X
                      color_counts[X] = color_counts.get(X, 0) + num_cells
                      # Add to the total count of non-zero painted cells
                      total_painted_non_zero += num_cells
                   # If X == 0, these cells contribute to final color 0 count. Handled later.
            # else: Column A was already finalized by a later operation. Skip this one.

    # Calculate the final count for color 0.
    # This includes cells initially 0 and never painted,
    # plus cells whose last painting operation used color 0.
    # Total cells = H * W. Cells ending with non-zero color = total_painted_non_zero.
    # Remaining cells must be color 0.
    count_0 = H * W - total_painted_non_zero
    
    # If there are any cells with color 0 in the final grid
    if count_0 > 0:
        # Add/update the count for color 0 in the dictionary.
        # The calculation `H * W - total_painted_non_zero` gives the total count for color 0.
        # There's no need to add to an existing count; just assign this value.
        color_counts[0] = count_0

    # Get a sorted list of color numbers that are present in the final grid
    # The keys in color_counts dictionary represent exactly these colors.
    # The problem requires output sorted by color number.
    colors = sorted(color_counts.keys())
    
    # Prepare output strings for potentially faster printing
    output_lines = []
    # First line: Number of distinct colors present (K)
    output_lines.append(str(len(colors)))
    # Subsequent lines: Color number and count for each color
    for c in colors:
        # Use f-string for formatted output: "color_number count"
        output_lines.append(f"{c} {color_counts[c]}")
    
    # Print all output lines joined by newlines. Add a final newline.
    # Using sys.stdout.write might be slightly faster than multiple print() calls for large outputs.
    sys.stdout.write("
".join(output_lines) + "
")

# Run the solution function
solve()