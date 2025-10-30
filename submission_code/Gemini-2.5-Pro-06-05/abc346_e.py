import sys
import collections

def main():
    """
    Solves the grid painting problem by processing operations in reverse.
    The core idea is that the final color of any cell is determined by the last
    operation that affects it. By processing operations in reverse order,
    the first time we encounter an operation for a specific row or column,
    we can determine its final contribution to the color counts, because any
    earlier operations on it are irrelevant.
    """
    
    # Read problem parameters: Height, Width, number of operations
    try:
        line = sys.stdin.readline()
        if not line.strip(): return # Handle empty input case
        H, W, M = map(int, line.split())
    except (IOError, ValueError):
        return

    # Read all M operations and store them in a list.
    ops = []
    for _ in range(M):
        ops.append(list(map(int, sys.stdin.readline().split())))

    # A dictionary to store the final counts of each color.
    # Using collections.defaultdict(int) simplifies adding counts.
    color_counts = collections.defaultdict(int)

    # Sets to keep track of rows and columns whose final color has been determined.
    painted_rows = set()
    painted_cols = set()

    # Process operations in reverse chronological order.
    for t, a, x in reversed(ops):
        if t == 1:  # Row painting operation
            # If this row's final color has already been determined by a later operation,
            # this one is completely overwritten, so we skip it.
            if a in painted_rows:
                continue

            # This is the last (and therefore final) paint on this row.
            # It successfully paints cells in columns that will not be painted over later.
            num_cells_painted = W - len(painted_cols)

            if num_cells_painted > 0:
                color_counts[x] += num_cells_painted
            
            # Mark this row as having its final color determined.
            painted_rows.add(a)

        else:  # t == 2, Column painting operation
            # If this column's final color has already been determined, skip.
            if a in painted_cols:
                continue
            
            # This is the final paint on this column.
            # It successfully paints cells in rows that will not be painted over later.
            num_cells_painted = H - len(painted_rows)
            
            if num_cells_painted > 0:
                color_counts[x] += num_cells_painted

            # Mark this column as having its final color determined.
            painted_cols.add(a)

    # Calculate the number of cells that were never painted.
    # These retain their initial color of 0.
    total_painted_count = sum(color_counts.values())
    unpainted_cell_count = H * W - total_painted_count

    # Add the count of these unpainted cells to the count for color 0.
    if unpainted_cell_count > 0:
        color_counts[0] += unpainted_cell_count

    # Prepare the output.
    # Filter out colors with a zero count.
    final_results = []
    for color, count in color_counts.items():
        if count > 0:
            final_results.append((color, count))
    
    # Sort the results by color in ascending order.
    final_results.sort()

    # Print the output in the required format.
    print(len(final_results))
    for color, count in final_results:
        print(f"{color} {count}")

if __name__ == "__main__":
    main()