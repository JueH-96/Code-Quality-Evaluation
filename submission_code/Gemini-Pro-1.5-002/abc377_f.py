def solve():
    n, m = map(int, input().split())
    pieces = []
    for _ in range(m):
        pieces.append(tuple(map(int, input().split())))

    threatened = set()
    for r, c in pieces:
        for i in range(1, n + 1):
            threatened.add((r, i))
            threatened.add((i, c))
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if r + c == i + j:
                    threatened.add((i, j))
                if r - c == i - j:
                    threatened.add((i, j))

    count = 0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if (i, j) not in threatened:
                count += 1
    print(count)

solve()