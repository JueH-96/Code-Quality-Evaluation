def solve():
    a, b, c, d, e, f = map(int, input().split())
    g, h, i, j, k, l = map(int, input().split())

    x1 = max(a, g)
    y1 = max(b, h)
    z1 = max(c, i)
    x2 = min(d, j)
    y2 = min(e, k)
    z2 = min(f, l)

    if x1 < x2 and y1 < y2 and z1 < z2:
        print("Yes")
    else:
        print("No")

solve()