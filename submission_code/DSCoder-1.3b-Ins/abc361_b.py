# YOUR CODE HERE

a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

def intersect(a, b, c, d, e, f, g, h, i, j, k, l):
    return max(0, min(d, e) - max(a, g)) * max(0, min(f, h) - max(c, i)) * max(0, min(l, k) - max(b, j))

if intersect(a, b, c, d, e, f, g, h, i, j, k, l) > 0:
    print('Yes')
else:
    print('No')