def solve():
    H, W, M = map(int, input().split())
    operations = []
    for _ in range(M):
        operations.append(list(map(int, input().split())))

    grid = [[0] * W for _ in range(H)]

    for T, A, X in operations:
        if T == 1:
            for j in range(W):
                grid[A-1][j] = X
        elif T == 2:
            for i in range(H):
                grid[i][A-1] = X

    color_counts = {}
    for i in range(H):
        for j in range(W):
            color = grid[i][j]
            if color not in color_counts:
                color_counts[color] = 0
            color_counts[color] += 1

    result = []
    for color, count in color_counts.items():
        if count > 0:
            result.append((color, count))

    result.sort()

    print(len(result))
    for color, count in result:
        print(color, count)

solve()