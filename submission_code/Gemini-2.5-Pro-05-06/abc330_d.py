import sys

def solve():
    N = int(sys.stdin.readline())
    grid = []
    for _ in range(N):
        # Read line and remove trailing newline character
        grid.append(sys.stdin.readline().strip())

    # Precompute the number of 'o's in each row and column
    row_o_counts = [0] * N
    col_o_counts = [0] * N

    for r in range(N):
        for c in range(N):
            if grid[r][c] == 'o':
                row_o_counts[r] += 1
                col_o_counts[c] += 1
    
    total_triples = 0
    # Iterate over each cell to consider it as the "corner" of an L-shape
    for r_corner in range(N):
        for c_corner in range(N):
            # The corner cell itself must be 'o'
            if grid[r_corner][c_corner] == 'o':
                
                # Number of other 'o's in the same row as the corner cell.
                # These are candidates for forming the horizontal arm of the L-shape.
                # (r_corner, c_other) where c_other != c_corner.
                # If row_o_counts[r_corner] is 1 (only the corner cell itself is 'o'),
                # then num_others_in_row will be 0.
                num_others_in_row = row_o_counts[r_corner] - 1
                
                # Number of other 'o's in the same column as the corner cell.
                # These are candidates for forming the vertical arm of the L-shape.
                # (r_other, c_corner) where r_other != r_corner.
                # If col_o_counts[c_corner] is 1, num_others_in_col will be 0.
                num_others_in_col = col_o_counts[c_corner] - 1
                
                # Each choice of an 'o' from the same row (but different column)
                # and an 'o' from the same column (but different row)
                # forms a unique L-shape triple with (r_corner, c_corner) as the corner.
                # The number of such triples is the product.
                # If num_others_in_row is 0 or num_others_in_col is 0,
                # then this product will be 0, correctly adding nothing to total_triples.
                total_triples += num_others_in_row * num_others_in_col
                    
    print(total_triples)

if __name__ == '__main__':
    solve()