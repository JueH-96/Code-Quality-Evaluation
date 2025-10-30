def main():
    N, M = map(int, input().split())
    
    # Read positions of existing pieces
    pieces_set = set()
    for _ in range(M):
        a, b = map(int, input().split())
        pieces_set.add((a, b))
    
    # Track all unsafe squares (where we can't place our piece)
    unsafe_set = set(pieces_set)  # Can't place on occupied squares
    
    # For each existing piece, mark the squares it can capture as unsafe
    for a, b in pieces_set:
        # All knight's moves from current piece position
        for di, dj in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
            ni, nj = a + di, b + dj
            # Check if the square is within the grid
            if 1 <= ni <= N and 1 <= nj <= N:
                unsafe_set.add((ni, nj))
    
    # Total safe squares = total squares - unsafe squares
    safe_squares = N * N - len(unsafe_set)
    
    print(safe_squares)

if __name__ == "__main__":
    main()