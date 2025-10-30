def solve():
    h, w, m = map(int, input().split())
    grid = [[0] * w for _ in range(h)]
    
    for _ in range(m):
        t, a, x = map(int, input().split())
        if t == 1:
            for j in range(w):
                grid[a - 1][j] = x
        else:
            for i in range(h):
                grid[i][a - 1] = x
                
    color_counts = {}
    for row in grid:
        for cell in row:
            if cell not in color_counts:
                color_counts[cell] = 0
            color_counts[cell] += 1
            
    sorted_colors = sorted(color_counts.keys())
    
    print(len(sorted_colors))
    for color in sorted_colors:
        print(color, color_counts[color])

solve()