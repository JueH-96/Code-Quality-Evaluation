def solve():
    n, m = map(int, input().split())
    occupied_squares = set()
    for _ in range(m):
        r, c = map(int, input().split())
        occupied_squares.add((r, c))

    attacked_squares = set()
    for r, c in occupied_squares:
        capturing_squares = [
            (r + 2, c + 1), (r + 1, c + 2), (r - 1, c + 2), (r - 2, c + 1),
            (r - 2, c - 1), (r - 1, c - 2), (r + 1, c - 2), (r + 2, c - 1)
        ]
        for ar, ac in capturing_squares:
            if 1 <= ar <= n and 1 <= ac <= n:
                attacked_squares.add((ar, ac))

    safe_squares_count = 0
    for r in range(1, n + 1):
        for c in range(1, n + 1):
            if (r, c) not in occupied_squares:
                is_safe = True
                for orow, ocol in occupied_squares:
                    if (orow + 2 == r and ocol + 1 == c) or \
                       (orow + 1 == r and ocol + 2 == c) or \
                       (orow - 1 == r and ocol + 2 == c) or \
                       (orow - 2 == r and ocol + 1 == c) or \
                       (orow - 2 == r and ocol - 1 == c) or \
                       (orow - 1 == r and ocol - 2 == c) or \
                       (orow + 1 == r and ocol - 2 == c) or \
                       (orow + 2 == r and ocol - 1 == c):
                        is_safe = False
                        break
                if is_safe:
                    safe_squares_count += 1

    print(safe_squares_count)

solve()