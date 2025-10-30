def solve():
    n, m = map(int, input().split())
    info = []
    for _ in range(m):
        info.append(list(map(int, input().split())))

    coords = {}
    coords[1] = (0, 0)
    
    undecided = set(range(1, n + 1))
    undecided.remove(1)
    
    changed = True
    while changed:
        changed = False
        for a, b, x, y in info:
            if a in coords and b in undecided:
                coords[b] = (coords[a][0] + x, coords[a][1] + y)
                undecided.remove(b)
                changed = True
            elif b in coords and a in undecided:
                coords[a] = (coords[b][0] - x, coords[b][1] - y)
                undecided.remove(a)
                changed = True

    for i in range(1, n + 1):
        if i in coords:
            print(coords[i][0], coords[i][1])
        else:
            print("undecidable")

solve()