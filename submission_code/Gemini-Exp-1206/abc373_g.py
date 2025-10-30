def cross_product(x1, y1, x2, y2):
    return x1 * y2 - x2 * y1

def intersect(a, b, c, d):
    ta = cross_product(c[0] - d[0], c[1] - d[1], a[0] - c[0], a[1] - c[1])
    tb = cross_product(c[0] - d[0], c[1] - d[1], b[0] - c[0], b[1] - c[1])
    tc = cross_product(a[0] - b[0], a[1] - b[1], c[0] - a[0], c[1] - a[1])
    td = cross_product(a[0] - b[0], a[1] - b[1], d[0] - a[0], d[1] - a[1])
    return tc * td < 0 and ta * tb < 0

def solve():
    n = int(input())
    p = []
    q = []
    for _ in range(n):
        a, b = map(int, input().split())
        p.append((a, b))
    for _ in range(n):
        c, d = map(int, input().split())
        q.append((c, d))

    import itertools
    for perm in itertools.permutations(range(n)):
        valid = True
        segments = []
        for i in range(n):
            segments.append((p[i], q[perm[i]]))
        
        for i in range(n):
            for j in range(i + 1, n):
                if intersect(segments[i][0], segments[i][1], segments[j][0], segments[j][1]):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            print(*[x + 1 for x in perm])
            return

    print("-1")

solve()