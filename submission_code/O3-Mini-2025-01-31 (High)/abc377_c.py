def main():
    import sys
    data = sys.stdin.read().split()
    if not data: 
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # Set to store positions that are already occupied by a piece.
    pieces = set()
    for _ in range(M):
        a = int(next(it))
        b = int(next(it))
        pieces.add((a, b))
    
    # The piece moves like a knight. These are the 8 knight moves.
    moves = [(2, 1), (1, 2), (-1, 2), (-2, 1),
             (-2, -1), (-1, -2), (1, -2), (2, -1)]
    
    # Set to store squares that can be captured by an existing piece.
    attacked = set()
    
    for (a, b) in pieces:
        for dx, dy in moves:
            na, nb = a + dx, b + dy
            if 1 <= na <= N and 1 <= nb <= N:
                attacked.add((na, nb))
    
    # The banned squares are those that are either already occupied or can be captured.
    banned = pieces.union(attacked)
    
    # Total squares possible in an N x N grid.
    total_squares = N * N
    safe_squares = total_squares - len(banned)
    
    sys.stdout.write(str(safe_squares))

if __name__ == '__main__':
    main()