# YOUR CODE HERE
import sys

def solve():
    N = int(sys.stdin.readline())
    grid = [sys.stdin.readline().strip() for _ in range(N)]

    # Precompute row and column counts of 'o's
    row_counts = [0] * N
    col_counts = [0] * N

    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                row_counts[i] += 1
                col_counts[j] += 1

    total_triples = 0

    # Iterate through each cell (i, j) in the grid.
    # If the cell contains 'o', it can potentially be the "corner" cell
    # of a valid triple. A valid triple satisfying the conditions
    # has the structure ((r1, c1), (r1, c2), (r2, c1)) where r1 != r2 and c1 != c2,
    # and all three cells contain 'o'.
    # In this structure, the cell (r1, c1) is the unique cell that is in the
    # shared row (r1) AND in the shared column (c1). We iterate through all 'o' cells
    # and consider each as the potential "corner" cell (i, j).
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                # If (i, j) is the corner cell, the other two cells must be:
                # 1. Another 'o' cell in the same row i, say (i, k), where k != j.
                # 2. Another 'o' cell in the same column j, say (l, j), where l != i.

                # The number of 'o's in row i is row_counts[i]. Since (i, j) is 'o',
                # the number of *other* 'o's in row i is row_counts[i] - 1.
                # This is the number of choices for the second cell (i, k).
                num_choices_for_k = row_counts[i] - 1
                
                # The number of 'o's in column j is col_counts[j]. Since (i, j) is 'o',
                # the number of *other* 'o's in column j is col_counts[j] - 1.
                # This is the number of choices for the third cell (l, j).
                num_choices_for_l = col_counts[j] - 1

                # To form a complete triple ((i, j), (i, k), (l, j)) that satisfies the conditions,
                # we need to be able to select at least one valid cell for (i, k) (i.e., num_choices_for_k >= 1)
                # AND at least one valid cell for (l, j) (i.e., num_choices_for_l >= 1).
                # The number of ways to choose one cell from the available options in the row
                # and one cell from the available options in the column is the product
                # of the number of choices: num_choices_for_k * num_choices_for_l.
                
                # If either num_choices_for_k < 1 or num_choices_for_l < 1, the product will be 0,
                # correctly indicating that no such triples can be formed with (i, j) as the corner.
                # Since grid[i][j] == 'o', we know row_counts[i] >= 1 and col_counts[j] >= 1,
                # so num_choices_for_k >= 0 and num_choices_for_l >= 0.
                # The product is non-zero only when both factors are positive.
                
                if num_choices_for_k >= 1 and num_choices_for_l >= 1:
                     total_triples += num_choices_for_k * num_choices_for_l
                # Alternatively, we could directly add the product (which is 0 if either count < 2):
                # total_triples += (row_counts[i] - 1) * (col_counts[j] - 1)
                # The explicit check `if ... >= 1` makes the requirement clearer.


    print(total_triples)

solve()