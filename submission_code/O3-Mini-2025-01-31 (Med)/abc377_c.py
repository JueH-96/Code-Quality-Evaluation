def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    # Read grid size and number of pieces
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # Knight move offsets (capture moves)
    moves = [(2, 1), (1, 2), (-1, 2), (-2, 1),
             (-2, -1), (-1, -2), (1, -2), (2, -1)]
    
    unsafe = set()
    
    # Process each existing piece
    for _ in range(M):
        a = int(next(it))
        b = int(next(it))
        # Add the square where the piece is already placed (occupied)
        unsafe.add((a, b))
        # Add all squares that can be captured by this piece
        for dx, dy in moves:
            ni, nj = a + dx, b + dy
            if 1 <= ni <= N and 1 <= nj <= N:
                unsafe.add((ni, nj))
    
    # Total number of squares
    total_squares = N * N
    # The safe squares are those that are empty and not attacked
    safe_count = total_squares - len(unsafe)
    
    sys.stdout.write(str(safe_count))


if __name__ == '__main__':
    main()