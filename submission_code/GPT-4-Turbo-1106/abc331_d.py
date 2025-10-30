def count_black_squares(N, P, queries):
    # Precompute the number of black squares in each NxN block
    black_count = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(N):
        for j in range(N):
            black_count[i + 1][j + 1] = black_count[i][j + 1] + black_count[i + 1][j] - black_count[i][j] + (P[i][j] == 'B')

    # Function to count black squares in a given rectangle
    def query_black_count(A, B, C, D):
        total_black = 0
        # Calculate number of complete NxN blocks and the remaining partial block
        full_blocks_row = (C - A) // N
        full_blocks_col = (D - B) // N
        partial_block_row = (C - A) % N
        partial_block_col = (D - B) % N
        
        # Add black squares from complete NxN blocks
        total_black += full_blocks_row * full_blocks_col * black_count[N][N]
        
        # Add black squares from partial blocks along the right and bottom edges
        if partial_block_row > 0:
            total_black += full_blocks_col * (black_count[partial_block_row][N] - black_count[0][N])
        if partial_block_col > 0:
            total_black += full_blocks_row * (black_count[N][partial_block_col] - black_count[N][0])
        
        # Add black squares from the remaining corner partial block
        if partial_block_row > 0 and partial_block_col > 0:
            total_black += black_count[partial_block_row][partial_block_col] - black_count[0][partial_block_col] - black_count[partial_block_row][0] + black_count[0][0]
        
        # Adjust for the top-left corner not being at (0, 0)
        top_left_row = A % N
        top_left_col = B % N
        total_black += black_count[top_left_row + partial_block_row][top_left_col + partial_block_col]
        total_black -= black_count[top_left_row][top_left_col + partial_block_col]
        total_black -= black_count[top_left_row + partial_block_row][top_left_col]
        total_black += black_count[top_left_row][top_left_col]
        
        return total_black

    # Process each query and return the result
    return [query_black_count(A, B, C + 1, D + 1) for A, B, C, D in queries]

# Read input
N, Q = map(int, input().split())
P = [input() for _ in range(N)]
queries = [tuple(map(int, input().split())) for _ in range(Q)]

# Solve the problem
results = count_black_squares(N, P, queries)

# Write output
for result in results:
    print(result)