def solve():
    n = int(input())
    grid = [input() for _ in range(n)]
    
    rows = []
    cols = []
    for i in range(n):
        row_count = 0
        col_count = 0
        for j in range(n):
            if grid[i][j] == 'o':
                row_count += 1
            if grid[j][i] == 'o':
                col_count += 1
        rows.append(row_count)
        cols.append(col_count)
    
    ans = 0
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 'o':
                ans += (rows[r] - 1) * (cols[c] - 1)
                
    print(ans)

solve()