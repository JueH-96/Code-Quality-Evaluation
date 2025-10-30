def repaint_grid(H, W, M, operations):
    row_colors = [0] * (H + 1)
    col_colors = [0] * (W + 1)
    row_updates = [0] * (H + 1)
    col_updates = [0] * (W + 1)
    
    for T, A, X in operations:
        if T == 1:
            row_colors[A] = X
            row_updates[A] += 1
        elif T == 2:
            col_colors[A] = X
            col_updates[A] += 1

    color_count = {}
    
    for r in range(1, H + 1):
        for c in range(1, W + 1):
            color = row_colors[r] if row_updates[r] > 0 else col_colors[c]
            if color in color_count:
                color_count[color] += 1
            else:
                color_count[color] = 1

    result = sorted(color_count.items())
    
    print(len(result))
    for color, count in result:
        print(color, count)

import sys
input = sys.stdin.read
data = input().splitlines()

H, W, M = map(int, data[0].split())
operations = [tuple(map(int, line.split())) for line in data[1:M + 1]]

repaint_grid(H, W, M, operations)