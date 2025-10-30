def solve():
    n, m = map(int, input().split())
    pieces = []
    for _ in range(m):
        pieces.append(tuple(map(int, input().split())))

    captured = set()
    for r, c in pieces:
        moves = [
            (r + 2, c + 1),
            (r + 1, c + 2),
            (r - 1, c + 2),
            (r - 2, c + 1),
            (r - 2, c - 1),
            (r - 1, c - 2),
            (r + 1, c - 2),
            (r + 2, c - 1),
        ]
        for nr, nc in moves:
            if 1 <= nr <= n and 1 <= nc <= n:
                captured.add((nr, nc))
    
    total_squares = n * n
    
    
    safe_squares = total_squares - len(captured) - m
    
    print(safe_squares)

solve()