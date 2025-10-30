a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

x_intersection = max(0, min(d, j) - max(a, g))
y_intersection = max(0, min(e, k) - max(b, h))
z_intersection = max(0, min(f, l) - max(c, i))

if x_intersection * y_intersection * z_intersection > 0:
    print('Yes')
else:
    print('No')