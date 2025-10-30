def solve():
    n, m = map(int, input().split())
    colored = []
    for _ in range(m):
        x, y, c = input().split()
        colored.append((int(x), int(y), c))

    row_max_b = [0] * (n + 1)
    row_min_w = [n + 1] * (n + 1)
    col_max_b = [0] * (n + 1)
    col_min_w = [n + 1] * (n + 1)

    for x, y, c in colored:
        if c == 'B':
            row_max_b[x] = max(row_max_b[x], y)
            col_max_b[y] = max(col_max_b[y], x)
        else:
            row_min_w[x] = min(row_min_w[x], y)
            col_min_w[y] = min(col_min_w[y], x)

    for i in range(1, n + 1):
        if row_max_b[i] >= row_min_w[i]:
            print("No")
            return

    for i in range(1, n + 1):
        if col_max_b[i] >= col_min_w[i]:
            print("No")
            return
    
    row_limit = [0] * (n + 1)
    col_limit = [0] * (n + 1)

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if row_max_b[i] < j < row_min_w[i] and col_max_b[j] < i < col_min_w[j]:
                row_limit[i] = j
                col_limit[j] = i
                break
    
    for i in range(1, n+1):
        if row_limit[i] == 0:
            row_limit[i] = row_min_w[i]

    for i in range(1, n+1):
        if col_limit[i] == 0:
            col_limit[i] = col_min_w[i]
            
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if row_max_b[i] < j < row_min_w[i] and col_max_b[j] < i < col_min_w[j]:
                if row_limit[i] != j or col_limit[j] != i:
                    print("No")
                    return

    print("Yes")


solve()