from collections import defaultdict

N = int(input())
XYZ = [tuple(map(int, input().split())) for _ in range(N)]

def solve(XYZ):
    d = defaultdict(int)
    for x1, y1, z1, x2, y2, z2 in XYZ:
        d[(x1, y1, z1)] += 1
        d[(x2, y1, z1)] += 1
        d[(x1, y2, z1)] += 1
        d[(x1, y1, z2)] += 1
        d[(x2, y2, z1)] += 1
        d[(x2, y1, z2)] += 1
        d[(x1, y2, z2)] += 1
        d[(x2, y2, z2)] += 1

    for x1, y1, z1, x2, y2, z2 in XYZ:
        yield (
            d[(x1, y1, z1)] + d[(x2, y1, z1)] + d[(x1, y2, z1)] + d[(x1, y1, z2)]
            + d[(x2, y2, z1)] + d[(x2, y1, z2)] + d[(x1, y2, z2)] + d[(x2, y2, z2)]
            - 8
        )

ans = list(solve(XYZ))
print(*ans, sep='
')