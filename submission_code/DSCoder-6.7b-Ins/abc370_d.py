from collections import defaultdict

def solve():
    H, W, Q = map(int, input().split())
    rows = defaultdict(lambda: [0, W+1])
    cols = defaultdict(lambda: [0, H+1])
    walls = set()

    for _ in range(Q):
        r, c = map(int, input().split())
        if (r, c) in walls:
            walls.remove((r, c))
            for i in range(rows[r][0], rows[r][1]):
                if (r, i) not in walls:
                    walls.add((r, i))
            for i in range(cols[c][0], cols[c][1]):
                if (i, c) not in walls:
                    walls.add((i, c))
        else:
            walls.add((r, c))
            rows[r] = [min(rows[r][0], c), max(rows[r][1], c+1)]
            cols[c] = [min(cols[c][0], r), max(cols[c][1], r+1)]

    print(len(walls))

solve()