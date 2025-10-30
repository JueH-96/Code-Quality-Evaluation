# YOUR CODE HERE
a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

max_x = max(a, g)
min_x = min(d, j)
max_y = max(b, h)
min_y = min(e, k)
max_z = max(c, i)
min_z = min(f, l)

if min_x > max_x or min_y > max_y or min_z > max_z:
    print("No")
else:
    print("Yes")