def solve():
    h, w, m = map(int, input().split())
    grid = [[0] * w for _ in range(h)]
    row_ops = {}
    col_ops = {}
    
    for _ in range(m):
        t, a, x = map(int, input().split())
        if t == 1:
            row_ops[a - 1] = (x, _)
        else:
            col_ops[a - 1] = (x, _)
    
    for r in range(h):
        if r in row_ops:
            color, time = row_ops[r]
            for c in range(w):
                grid[r][c] = color
                
    for c in range(w):
        if c in col_ops:
            color, time = col_ops[c]
            for r in range(h):
                grid[r][c] = color

    counts = {}
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            counts[color] = counts.get(color, 0) + 1
    
    sorted_counts = sorted(counts.items())
    
    print(len(sorted_counts))
    for color, count in sorted_counts:
        print(color, count)

solve()