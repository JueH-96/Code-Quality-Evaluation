# YOUR CODE HERE

def check_intersection(a, b, c, d, e, f, g, h, i, j, k, l):
    x_overlap = max(0, min(d, j) - max(a, g))
    y_overlap = max(0, min(e, k) - max(b, h))
    z_overlap = max(0, min(f, l) - max(c, i))
    return x_overlap * y_overlap * z_overlap > 0

a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

if check_intersection(a, b, c, d, e, f, g, h, i, j, k, l):
    print("Yes")
else:
    print("No")