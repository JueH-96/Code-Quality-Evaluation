def solve():
    n = int(input())
    grid = [input() for _ in range(n)]
    
    count = 0
    for r1 in range(n):
        for c1 in range(n):
            if grid[r1][c1] == 'o':
                for r2 in range(n):
                    for c2 in range(n):
                        if grid[r2][c2] == 'o' and (r1 != r2 or c1 != c2):
                            for r3 in range(n):
                                for c3 in range(n):
                                    if grid[r3][c3] == 'o' and ((r1 != r3 or c1 != c3) and (r2 != r3 or c2 != c3)):
                                        cells = [(r1, c1), (r2, c2), (r3, c3)]
                                        
                                        rows = [r1, r2, r3]
                                        cols = [c1, c2, c3]
                                        
                                        row_counts = {}
                                        for r in rows:
                                            row_counts[r] = row_counts.get(r, 0) + 1
                                        
                                        col_counts = {}
                                        for c in cols:
                                            col_counts[c] = col_counts.get(c, 0) + 1
                                        
                                        row_two = False
                                        for r in row_counts:
                                            if row_counts[r] == 2:
                                                row_two = True
                                                break
                                        
                                        col_two = False
                                        for c in col_counts:
                                            if col_counts[c] == 2:
                                                col_two = True
                                                break
                                        
                                        if row_two and col_two:
                                            count += 1
    print(count // 6)

solve()