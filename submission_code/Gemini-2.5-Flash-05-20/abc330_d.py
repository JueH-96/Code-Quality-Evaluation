import sys

def solve():
    N = int(sys.stdin.readline())
    
    grid = []
    for _ in range(N):
        grid.append(sys.stdin.readline().strip())
    
    # Initialize lists to store counts of 'o's in each row and column
    row_counts = [0] * N
    col_counts = [0] * N
    
    # Precompute counts of 'o's for all rows and columns
    for r in range(N):
        for c in range(N):
            if grid[r][c] == 'o':
                row_counts[r] += 1
                col_counts[c] += 1
                
    total_triples = 0
    
    # Iterate through each cell (r, c) in the grid
    # A cell (r, c) that contains 'o' can be the "pivot" cell of a valid triple.
    for r in range(N):
        for c in range(N):
            if grid[r][c] == 'o':
                # 'count_row_partners' is the number of other 'o's in the same row 'r'.
                # These are potential candidates for the second cell of the triple, (r, c_other_c).
                count_row_partners = row_counts[r] - 1
                
                # 'count_col_partners' is the number of other 'o's in the same column 'c'.
                # These are potential candidates for the third cell of the triple, (r_other_r, c).
                count_col_partners = col_counts[c] - 1
                
                # To form a valid triple with (r, c) as the pivot,
                # there must be at least one other 'o' in its row
                # AND at least one other 'o' in its column.
                # If either count is zero, no such triple can be formed with (r,c) as pivot.
                if count_row_partners > 0 and count_col_partners > 0:
                    # The number of ways to pick one other 'o' from the row
                    # multiplied by the number of ways to pick one other 'o' from the column
                    # gives the total number of distinct triples for which (r,c) is the pivot.
                    total_triples += count_row_partners * count_col_partners
                    
    print(total_triples)

# Ensure the solve function is called when the script is executed
if __name__ == '__main__':
    solve()