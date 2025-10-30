def solve():
    n, m = map(int, input().split())
    pieces = []
    for _ in range(m):
        pieces.append(tuple(map(int, input().split())))
    
    captured = set()
    
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            
            can_be_captured = False
            for a, b in pieces:
                if (a + 2 == i and b + 1 == j) or \
                   (a + 1 == i and b + 2 == j) or \
                   (a - 1 == i and b + 2 == j) or \
                   (a - 2 == i and b + 1 == j) or \
                   (a - 2 == i and b - 1 == j) or \
                   (a - 1 == i and b - 2 == j) or \
                   (a + 1 == i and b - 2 == j) or \
                   (a + 2 == i and b - 1 == j):
                    can_be_captured = True
                    break
            
            if can_be_captured:
                captured.add((i, j))
    
    existing_pieces_set = set(pieces)
    
    safe_squares = 0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if (i, j) not in existing_pieces_set and (i, j) not in captured:
                safe_squares += 1
    
    print(safe_squares)

solve()