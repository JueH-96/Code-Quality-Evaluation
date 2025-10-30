# YOUR CODE HERE
n = int(input())
grid = []
for _ in range(n):
    grid.append(input().strip())

total = 0

# For each cell that could be the corner of the L
for i in range(n):
    for j in range(n):
        if grid[i][j] == 'o':
            # Count 'o's in the same row (excluding current cell)
            row_count = 0
            for k in range(n):
                if k != j and grid[i][k] == 'o':
                    row_count += 1
            
            # Count 'o's in the same column (excluding current cell)
            col_count = 0
            for k in range(n):
                if k != i and grid[k][j] == 'o':
                    col_count += 1
            
            # Number of L-shapes with corner at (i, j)
            total += row_count * col_count

print(total)