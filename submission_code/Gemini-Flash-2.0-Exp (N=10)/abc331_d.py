def solve():
    n, q = map(int, input().split())
    grid = [input() for _ in range(n)]
    
    def count_blacks(r1, c1, r2, c2):
        count = 0
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if grid[r % n][c % n] == 'B':
                    count += 1
        return count

    for _ in range(q):
        a, b, c, d = map(int, input().split())
        
        total_blacks = 0
        
        
        rows = c - a + 1
        cols = d - b + 1
        
        full_row_blocks = rows // n
        full_col_blocks = cols // n
        
        remaining_rows = rows % n
        remaining_cols = cols % n
        
        
        total_blacks += full_row_blocks * full_col_blocks * count_blacks(0,0,n-1,n-1)
        
        total_blacks += full_row_blocks * count_blacks(0,0,n-1,remaining_cols-1)
        
        total_blacks += full_col_blocks * count_blacks(0,0,remaining_rows-1,n-1)
        
        total_blacks += count_blacks(0,0,remaining_rows-1,remaining_cols-1)
        
        
        
        
        
        
        print(total_blacks)

solve()