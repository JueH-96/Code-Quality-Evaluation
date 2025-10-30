def solve():
    n = int(input())
    p = list(map(int, input().split()))
    q = list(map(int, input().split()))

    grid = [['0'] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if p[i] < q[j]:
                grid[i][j] = '0'
            else:
                grid[i][j] = '1'
    
    p_inv = [0] * (n + 1)
    q_inv = [0] * (n + 1)
    for i in range(n):
        p_inv[p[i]] = i
        q_inv[q[i]] = i
    
    
    def check_rows(grid, p):
        rows = []
        for i in range(n):
            rows.append("".join(grid[i]))
        
        for i in range(n - 1):
            if rows[p_inv[i+1]+1-1] >= rows[p_inv[i+2]+1-1]:
                return False
        return True

    def check_cols(grid, q):
        cols = []
        for j in range(n):
            col = ""
            for i in range(n):
                col += grid[i][j]
            cols.append(col)
        
        for i in range(n - 1):
            if cols[q_inv[i+1]+1-1] >= cols[q_inv[i+2]+1-1]:
                return False
        return True

    for row in grid:
        print("".join(row))

solve()