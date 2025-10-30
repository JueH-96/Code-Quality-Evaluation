def count_triples(grid, N):
    # Precompute the number of 'o's in each row and column
    row_counts = [0] * N
    col_counts = [0] * N
    
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                row_counts[i] += 1
                col_counts[j] += 1
    
    count = 0
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                # For cell (i, j), number of 'o's in the same row excluding (i, j)
                same_row_count = row_counts[i] - 1
                
                # For cell (i, j), number of 'o's in the same column excluding (i, j)
                same_col_count = col_counts[j] - 1
                
                # For each combination of a cell in the same row and a cell in the same column,
                # we have a valid triple.
                count += same_row_count * same_col_count
    
    return count

N = int(input())
grid = []
for _ in range(N):
    row = input().strip()
    grid.append(row)

print(count_triples(grid, N))