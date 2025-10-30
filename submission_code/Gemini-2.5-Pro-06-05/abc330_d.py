import sys

def main():
    """
    Reads the grid, calculates the number of valid triples, and prints the result.
    """
    # Read input from stdin for efficiency
    try:
        N = int(sys.stdin.readline())
        S = [sys.stdin.readline().strip() for _ in range(N)]
    except (ValueError, IndexError):
        # Handle empty input
        return

    # Pre-compute the number of 'o's in each row and column
    row_counts = [s.count('o') for s in S]
    col_counts = [0] * N
    for i in range(N):
        for j in range(N):
            if S[i][j] == 'o':
                col_counts[j] += 1

    # Calculate the number of triples
    total_triples = 0
    for i in range(N):
        for j in range(N):
            if S[i][j] == 'o':
                # Consider the cell (i, j) as the corner of an 'L' shape.
                # We need one other 'o' in the same row and one other 'o' in the same column.
                
                # Number of other 'o's to choose from in row i
                num_in_row = row_counts[i] - 1
                
                # Number of other 'o's to choose from in column j
                num_in_col = col_counts[j] - 1
                
                # The number of triples with (i, j) as the corner is the product.
                # This is only non-zero if there are partners in both the row and column.
                if num_in_row > 0 and num_in_col > 0:
                    total_triples += num_in_row * num_in_col

    # Print the final count
    print(total_triples)

if __name__ == "__main__":
    main()