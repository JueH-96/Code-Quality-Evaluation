def solve():
    n = int(input())
    grid = [input() for _ in range(n)]
    
    o_cells = []
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 'o':
                o_cells.append((r, c))
    
    count = 0
    for i in range(len(o_cells)):
        for j in range(i + 1, len(o_cells)):
            for k in range(j + 1, len(o_cells)):
                r1, c1 = o_cells[i]
                r2, c2 = o_cells[j]
                r3, c3 = o_cells[k]
                
                rows = {r1, r2, r3}
                cols = {c1, c2, c3}
                
                if len(rows) == 2 and len(cols) == 2:
                    count += 1
    print(count)

solve()