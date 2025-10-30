def solve():
    a, b, c, d, e, f = map(int, input().split())
    g, h, i, j, k, l = map(int, input().split())

    x_overlap = max(0, min(d, j) - max(a, g))
    y_overlap = max(0, min(e, k) - max(b, h))
    z_overlap = max(0, min(f, l) - max(c, i))

    if x_overlap > 0 and y_overlap > 0 and z_overlap > 0:
        print("Yes")
    else:
        print("No")

solve()