# YOUR CODE HERE
def solve():
    h, w, m = map(int, input().split())
    ops = []
    for _ in range(m):
        ops.append(list(map(int, input().split())))

    grid = [[0] * w for _ in range(h)]

    for op in ops:
        t, a, x = op
        if t == 1:
            grid[a - 1] = [x] * w
        else:
            for i in range(h):
                grid[i][a - 1] = x

    color_counts = {}
    for row in grid:
        for cell in row:
            color_counts[cell] = color_counts.get(cell, 0) + 1

    result = []
    for color, count in color_counts.items():
        if count > 0:
            result.append((color, count))

    result.sort()

    print(len(result))
    for color, count in result:
        print(color, count)

solve()