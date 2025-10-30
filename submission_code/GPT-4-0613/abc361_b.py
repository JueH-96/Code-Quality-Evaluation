# YOUR CODE HERE
a, b, c, d, e, f = map(int, input().split())
g, h, i, j, k, l = map(int, input().split())

if max(a, g) < min(d, j) and max(b, h) < min(e, k) and max(c, i) < min(f, l):
    print('Yes')
else:
    print('No')