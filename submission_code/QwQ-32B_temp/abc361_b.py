a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

x_start = max(a, g)
x_end = min(d, j)
x_ok = x_start < x_end

y_start = max(b, h)
y_end = min(e, k)
y_ok = y_start < y_end

z_start = max(c, i)
z_end = min(f, l)
z_ok = z_start < z_end

if x_ok and y_ok and z_ok:
    print("Yes")
else:
    print("No")