# YOUR CODE HERE
H, W, M = map(int, input().split())
grid = [[0 for _ in range(W)] for _ in range(H)]
color_counts = {}
for i in range(M):
    T, A, X = map(int, input().split())
    if T == 1:
        for j in range(W):
            grid[A - 1][j] = X
    else:
        for j in range(H):
            grid[j][A - 1] = X
for row in grid:
    for color in row:
        if color in color_counts:
            color_counts[color] += 1
        else:
            color_counts[color] = 1
result = []
for color, count in sorted(color_counts.items()):
    result.append((color, count))
print(len(result))
for color, count in result:
    print(color, count)