# YOUR CODE HERE
import sys

def solve():
    # Read N from stdin
    N = int(sys.stdin.readline())
    
    # Read the N x N grid from stdin
    # Store the grid as a list of strings
    grid = [sys.stdin.readline().strip() for _ in range(N)]

    # Precompute the count of 'o's in each row
    row_counts = [0] * N
    # Precompute the count of 'o's in each column
    col_counts = [0] * N

    # Iterate through the grid once to compute both row and column counts
    # This is O(N^2)
    for r in range(N):
        for c in range(N):
            if grid[r][c] == 'o':
                # Increment count for row r
                row_counts[r] += 1
                # Increment count for column c
                col_counts[c] += 1

    # Initialize the total count of valid triples
    ans = 0
    
    # Iterate through each cell (r, c) of the grid again
    # This is also O(N^2)
    for r in range(N):
        for c in range(N):
            # Check if the current cell contains 'o'.
            # Only cells with 'o' can be part of a valid triple.
            # We consider each 'o' cell as a potential corner of an L-shape.
            if grid[r][c] == 'o':
                # Calculate the number of other 'o's in the same row r.
                # The cell (r, c) itself is included in row_counts[r], so we subtract 1 
                # to get the count of other 'o's in the row.
                # Let this be num_horizontal_neighbors.
                # To form the horizontal leg of an L-shape originating from (r, c), 
                # we need at least one other 'o' in the same row.
                # So, num_horizontal_neighbors must be > 0, which means row_counts[r] > 1.
                num_horizontal_neighbors = row_counts[r] - 1
                
                # Calculate the number of other 'o's in the same column c.
                # Similarly, subtract 1 from col_counts[c] to exclude the cell (r, c) itself.
                # Let this be num_vertical_neighbors.
                # To form the vertical leg of an L-shape originating from (r, c),
                # we need at least one other 'o' in the same column.
                # So, num_vertical_neighbors must be > 0, which means col_counts[c] > 1.
                num_vertical_neighbors = col_counts[c] - 1
                
                # A valid triple satisfying the conditions forms an L-shape.
                # The cell (r, c) acts as the "corner" of the L-shape if it shares its row
                # with exactly one other cell in the triple, and its column with exactly one other cell
                # in the triple.
                # This requires selecting one 'o' neighbor from the same row (r) but different column,
                # and one 'o' neighbor from the same column (c) but different row.
                
                # If there is at least one other 'o' in the row (num_horizontal_neighbors > 0)
                # AND at least one other 'o' in the column (num_vertical_neighbors > 0),
                # then cell (r, c) can be the corner of L-shaped triples.
                # The number of ways to choose one horizontal neighbor is num_horizontal_neighbors.
                # The number of ways to choose one vertical neighbor is num_vertical_neighbors.
                # Each combination of one horizontal neighbor (p, r) with r != c and one vertical neighbor (s, q) with s != r forms a unique valid triple: {(r, c), (r, neighbor_c), (neighbor_r, c)}.
                # The total number of such triples with corner (r, c) is the product of the number of choices.
                if num_horizontal_neighbors > 0 and num_vertical_neighbors > 0:
                     ans += num_horizontal_neighbors * num_vertical_neighbors

    # Print the final answer (total count of valid triples) to stdout
    print(ans)

# Call the solve function when the script is executed
solve()