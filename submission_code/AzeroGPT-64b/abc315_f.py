from math import sqrt

def dist(p, q): return sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
def between(a, b, c): return dist(a, b) + dist(b, c) >= dist(a, c)

n = int(input())
c = [tuple(map(int, input().split())) for _ in range(n)]

# Convex hull and Convex hull trick optimization
H = []
for p in c:
    while len(H) > 1 and between(H[-2], H[-1], p): H.pop()
    H.append(p)
H = H + H[:len(H)-1][::-1]

d = {p:[0, 0] for p in c + [(c[0], c[-1])]}

for i in range(1, len(H) - 1):
    p, q = H[i-1], H[i+1]
    c = (p[0] + q[0], p[1] + q[1])
    d[q] = [(d[q][0] + dist(p, q)) * 2, d[q][1] + 2 * (d[q][1] > 0) * dist(p, c)]

print(d[(c[0], c[-1])][0])