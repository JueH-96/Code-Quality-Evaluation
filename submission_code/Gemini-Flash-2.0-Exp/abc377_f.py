def solve():
    n, m = map(int, input().split())
    pieces = []
    for _ in range(m):
        pieces.append(tuple(map(int, input().split())))

    captured = set()
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            can_capture = False
            for a, b in pieces:
                if r == a or c == b or r + c == a + b or r - c == a - b:
                    can_capture = True
                    break
            if can_capture:
                captured.add((r, c))

    print(n * n - len(captured))

solve()