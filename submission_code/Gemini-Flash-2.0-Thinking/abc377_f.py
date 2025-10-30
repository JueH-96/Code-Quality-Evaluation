def solve():
    n, m = map(int, input().split())
    existing_pieces = []
    for _ in range(m):
        existing_pieces.append(tuple(map(int, input().split())))

    occupied = set(existing_pieces)
    safe_count = 0

    for r in range(1, n + 1):
        for c in range(1, n + 1):
            if (r, c) not in occupied:
                is_safe = True
                for er, ec in existing_pieces:
                    if r == er or c == ec or r + c == er + ec or r - c == er - ec:
                        is_safe = False
                        break
                if is_safe:
                    safe_count += 1
    print(safe_count)

solve()