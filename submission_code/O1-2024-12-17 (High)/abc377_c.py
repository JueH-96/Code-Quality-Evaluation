def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    coords = list(map(int, data[2:]))

    # Knight-like moves (relative positions)
    moves = [(2, 1), (1, 2), (-1, 2), (-2, 1),
             (-2, -1), (-1, -2), (1, -2), (2, -1)]

    # Keep track of all squares occupied or attacked by any of the M pieces
    attacked = set()

    # For each piece, record its square and the squares it can attack
    for i in range(M):
        r = coords[2*i]
        c = coords[2*i + 1]
        # The piece's own position
        attacked.add((r, c))
        # Squares it can attack (if they lie within the board)
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 1 <= nr <= N and 1 <= nc <= N:
                attacked.add((nr, nc))

    # Total squares is N*N, subtract the count of occupied/attacked squares
    total_squares = N * N  # up to 10^18, which Python can handle
    result = total_squares - len(attacked)

    print(result)

# Do not forget to call main()
if __name__ == "__main__":
    main()