def solve():
    n, m = map(int, input().split())
    pieces = []
    for _ in range(m):
        pieces.append(list(map(int, input().split())))

    captured = set()
    for r, c in pieces:
        for i in range(1, n + 1):
            captured.add((r, i))
            captured.add((i, c))
        for i in range(1, n + 1):
            j = r + c - i
            if 1 <= j <= n:
                captured.add((i, j))
            j = r - c + i
            if 1 <= j <= n:
                captured.add((i, j))
    
    print(n * n - len(captured))

solve()