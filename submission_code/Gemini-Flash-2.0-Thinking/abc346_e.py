def solve():
    h, w, m = map(int, input().split())
    operations = []
    for _ in range(m):
        operations.append(list(map(int, input().split())))

    grid = [[0 for _ in range(w)] for _ in range(h)]

    for op_type, a, color in operations:
        if op_type == 1:
            for j in range(w):
                grid[a - 1][j] = color
        else:
            for i in range(h):
                grid[i][a - 1] = color

    color_counts = {}
    for i in range(h):
        for j in range(w):
            color = grid[i][j]
            color_counts[color] = color_counts.get(color, 0) + 1

    distinct_colors = sorted(color_counts.keys())
    print(len(distinct_colors))
    for color in distinct_colors:
        print(color, color_counts[color])

solve()