def solve():
    n, m = map(int, input().split())
    photos = []
    for _ in range(m):
        photos.append(list(map(int, input().split())))

    pairs = set()
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            pairs.add((i, j))

    for photo in photos:
        for i in range(n - 1):
            p1 = photo[i]
            p2 = photo[i+1]
            if p1 > p2:
                p1, p2 = p2, p1
            if (p1, p2) in pairs:
                pairs.remove((p1, p2))

    print(len(pairs))

solve()