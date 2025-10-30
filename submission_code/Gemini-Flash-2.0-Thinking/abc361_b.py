a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

x_start = max(a, g)
x_end = min(d, j)
y_start = max(b, h)
y_end = min(e, k)
z_start = max(c, i)
z_end = min(f, l)

if x_start < x_end and y_start < y_end and z_start < z_end:
    print("Yes")
else:
    print("No")