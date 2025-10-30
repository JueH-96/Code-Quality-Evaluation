def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # Store the coordinates of the M pieces.
    occupied = set()
    pieces = []
    for _ in range(M):
        a = int(next(it))
        b = int(next(it))
        occupied.add((a, b))
        pieces.append((a, b))

    # Predefined moves that a piece can capture (like a knight in chess)
    moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    
    # Build a set of squares that are attacked by any of the M pieces.
    attacked = set()
    for a, b in pieces:
        for dx, dy in moves:
            x = a + dx
            y = b + dy
            if 1 <= x <= N and 1 <= y <= N:
                attacked.add((x, y))
    
    # Unsafe squares are those already occupied or are attacked by any piece.
    unsafe = occupied.union(attacked)
    
    # Total squares in an N x N grid.
    total = N * N
    # Safe squares are those that are empty and not attacked.
    safe = total - len(unsafe)
    
    sys.stdout.write(str(safe))

if __name__ == '__main__':
    main()