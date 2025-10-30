# YOUR CODE HERE
def can_color_grid(N, M, colored_cells):
    row_black = [0] * (N + 1)
    col_black = [0] * (N + 1)
    row_white = [N + 1] * (N + 1)
    col_white = [N + 1] * (N + 1)

    for x, y, color in colored_cells:
        if color == 'B':
            row_black[x] = max(row_black[x], y)
            col_black[y] = max(col_black[y], x)
        else:
            row_white[x] = min(row_white[x], y)
            col_white[y] = min(col_white[y], x)

    for i in range(1, N + 1):
        if row_black[i] >= row_white[i] or col_black[i] >= col_white[i]:
            return "No"

    for i in range(1, N + 1):
        for j in range(1, N + 1):
            if (i > col_black[j] and j > row_black[i]) or (i < col_white[j] and j < row_white[i]):
                return "No"

    return "Yes"

N, M = map(int, input().split())
colored_cells = []
for _ in range(M):
    x, y, c = input().split()
    colored_cells.append((int(x), int(y), c))

result = can_color_grid(N, M, colored_cells)
print(result)