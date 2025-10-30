def count_triples(N, grid):
    # Step 1: Create a list to store the positions of 'o's
    positions = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                positions.append((i, j))

    # Step 2: Count occurrences of 'o's in each row and column
    row_count = [0] * N
    col_count = [0] * N
    for i, j in positions:
        row_count[i] += 1
        col_count[j] += 1

    # Step 3: Calculate the number of valid triples
    total_triples = 0
    for i, j in positions:
        # For each 'o' at (i, j), we need to find pairs of 'o's in the same row and column
        # Count pairs in the same row
        pairs_in_row = row_count[i] - 1  # Exclude the current 'o'
        # Count pairs in the same column
        pairs_in_col = col_count[j] - 1  # Exclude the current 'o'
        
        # Each combination of a pair from the row and a pair from the column gives a valid triple
        total_triples += pairs_in_row * pairs_in_col

    return total_triples

import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
grid = data[1:N+1]

result = count_triples(N, grid)
print(result)