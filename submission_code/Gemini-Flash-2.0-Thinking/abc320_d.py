def solve():
    n, m = map(int, input().split())
    informations = []
    for _ in range(m):
        a, b, x, y = map(int, input().split())
        informations.append((a - 1, b - 1, x, y))

    coords = [None] * n
    coords[0] = (0, 0)

    while True:
        updated = False
        for a, b, x, y in informations:
            if coords[a] is not None and coords[b] is None:
                coords[b] = (coords[a][0] + x, coords[a][1] + y)
                updated = True
            elif coords[b] is not None and coords[a] is None:
                coords[a] = (coords[b][0] - x, coords[b][1] - y)
                updated = True
        if not updated:
            break

    for coord in coords:
        if coord is None:
            print("undecidable")
        else:
            print(coord[0], coord[1])

solve()