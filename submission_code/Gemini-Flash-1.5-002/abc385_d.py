n, m, sx, sy = map(int, input().split())
houses = set()
for i in range(n):
    x, y = map(int, input().split())
    houses.add((x, y))
x, y = sx, sy
passed = set()
for i in range(m):
    d, c = input().split()
    c = int(c)
    if d == 'U':
        if x in {h[0] for h in houses}:
            for j in range(min(y, y + c), max(y, y + c) + 1):
                if (x, j) in houses:
                    passed.add((x, j))
        y += c
    elif d == 'D':
        if x in {h[0] for h in houses}:
            for j in range(min(y, y - c), max(y, y - c) + 1):
                if (x, j) in houses:
                    passed.add((x, j))
        y -= c
    elif d == 'L':
        if y in {h[1] for h in houses}:
            for j in range(min(x, x - c), max(x, x - c) + 1):
                if (j, y) in houses:
                    passed.add((j, y))
        x -= c
    elif d == 'R':
        if y in {h[1] for h in houses}:
            for j in range(min(x, x + c), max(x, x + c) + 1):
                if (j, y) in houses:
                    passed.add((j, y))
        x += c
print(x, y, len(passed))