a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

overlap_x = max(0, min(d, j) - max(a, g))
overlap_y = max(0, min(e, k) - max(b, h))
overlap_z = max(0, min(f, l) - max(c, i))

if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
    print("Yes")
else:
    print("No")