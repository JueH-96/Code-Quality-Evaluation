def solve():
    n, m = map(int, input().split())
    pieces = []
    for _ in range(m):
        pieces.append(tuple(map(int, input().split())))

    threatened = set()
    for r, c in pieces:
        for dr, dc in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
            nr, nc = r + dr, c + dc
            if 1 <= nr <= n and 1 <= nc <= n:
                threatened.add((nr, nc))

    ans = n * n - m - len(threatened)
    print(ans)

solve()