a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

x_start = max(a, g)
x_end = min(d, j)
y_start = max(b, h)
y_end = min(e, k)
z_start = max(c, i)
z_end = min(f, l)

if (x_end > x_start) and (y_end > y_start) and (z_end > z_start):
    print("Yes")
else:
    print("No")