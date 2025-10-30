def solve():
    n, m = map(int, input().split())
    pieces = []
    for _ in range(m):
        pieces.append(tuple(map(int, input().split())))

    occupied = set(pieces)
    attacked = set()

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
                attacked.add((nr, nc))

    safe_count = 0
    
    min_r = float('inf')
    max_r = float('-inf')
    min_c = float('inf')
    max_c = float('-inf')

    for r,c in attacked:
        min_r = min(min_r, r)
        max_r = max(max_r, r)
        min_c = min(min_c, c)
        max_c = max(max_c, c)

    if not attacked:
        print(n*n)
        return

    
    for r in range(max(1,min_r-5), min(n+1, max_r+6)):
        for c in range(max(1, min_c-5), min(n+1, max_c+6)):
            if (r,c) not in occupied and (r,c) not in attacked:
                safe_count+=1

    print(n * n - len(occupied) - len(attacked) + len(occupied.intersection(attacked)))

solve()