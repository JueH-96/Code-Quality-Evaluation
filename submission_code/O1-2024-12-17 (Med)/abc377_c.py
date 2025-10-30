def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    
    pieces = []
    idx = 2
    for _ in range(M):
        r = int(data[idx])
        c = int(data[idx+1])
        pieces.append((r, c))
        idx += 2

    # Set of occupied squares (where existing pieces are)
    occupied = set(pieces)

    # Possible "knight-like" moves that can capture
    moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
    
    # Collect all threatened squares
    threatened = set()
    for (r, c) in pieces:
        for (dr, dc) in moves:
            nr, nc = r + dr, c + dc
            # Check board boundaries
            if 1 <= nr <= N and 1 <= nc <= N:
                threatened.add((nr, nc))
    
    # Count how many threatened squares are actually occupied,
    # because those can't be used anyway (they're not "empty")
    overlap = sum(1 for sq in occupied if sq in threatened)
    
    # Number of threatened empty squares
    threatened_empty = len(threatened) - overlap
    
    # Calculate the answer
    total_squares = N * N  # may be as large as 10^18
    answer = total_squares - M - threatened_empty
    
    print(answer)

# Call main() to execute the solution
if __name__ == "__main__":
    main()