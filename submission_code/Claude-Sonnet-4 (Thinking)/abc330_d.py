n = int(input())
grid = []
for _ in range(n):
    grid.append(input().strip())

count = 0
for r in range(n):
    for c in range(n):
        if grid[r][c] == 'o':
            # Count cells in same row but different column
            row_count = 0
            for c2 in range(n):
                if c2 != c and grid[r][c2] == 'o':
                    row_count += 1
            
            # Count cells in same column but different row  
            col_count = 0
            for r2 in range(n):
                if r2 != r and grid[r2][c] == 'o':
                    col_count += 1
            
            count += row_count * col_count

print(count)