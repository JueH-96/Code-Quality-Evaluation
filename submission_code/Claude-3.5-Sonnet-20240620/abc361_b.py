# YOUR CODE HERE
def intersect(a, b, c, d, e, f, g, h, i, j, k, l):
    return max(a, g) < min(d, j) and max(b, h) < min(e, k) and max(c, i) < min(f, l)

a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

if intersect(a, b, c, d, e, f, g, h, i, j, k, l):
    print("Yes")
else:
    print("No")