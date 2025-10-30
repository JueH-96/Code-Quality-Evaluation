def intersect(a, b, c, d, e, f, g, h, i, j, k, l):
    x1 = max(a, g)
    y1 = max(b, h)
    z1 = max(c, i)
    x2 = min(d, j)
    y2 = min(e, k)
    z2 = min(f, l)

    if x1 < x2 and y1 < y2 and z1 < z2:
        return (x2 - x1) * (y2 - y1) * (z2 - z1)
    else:
        return 0

a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

volume = intersect(a, b, c, d, e, f, g, h, i, j, k, l)

if volume > 0:
    print('Yes')
else:
    print('No')