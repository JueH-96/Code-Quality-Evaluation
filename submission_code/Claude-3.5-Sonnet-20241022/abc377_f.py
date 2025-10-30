def get_captured_squares(N, pieces):
    # Keep track of occupied rows, columns, diagonals (i+j) and anti-diagonals (i-j)
    rows = set()
    cols = set()
    diag = set()  # i+j
    anti_diag = set()  # i-j
    
    # Add all pieces' positions to respective sets
    for i, j in pieces:
        rows.add(i)
        cols.add(j)
        diag.add(i + j)
        anti_diag.add(i - j)
    
    # Total squares that can be captured
    captured = 0
    
    # For each position in grid
    for i in range(1, N+1):
        for j in range(1, N+1):
            # Skip if position already has a piece
            if (i, j) in pieces:
                continue
                
            # Check if position can be captured
            if (i in rows or 
                j in cols or 
                i + j in diag or 
                i - j in anti_diag):
                captured += 1
                
    # Return safe squares (total squares - captured squares - occupied squares)
    return N * N - captured - len(pieces)

def main():
    # Read input
    N, M = map(int, input().split())
    pieces = set()
    for _ in range(M):
        a, b = map(int, input().split())
        pieces.add((a, b))
    
    # For large N, use mathematical formula to avoid TLE
    if N > 10000:
        # Count unique rows, cols, diagonals, anti-diagonals
        rows = set(i for i, _ in pieces)
        cols = set(j for _, j in pieces)
        diag = set(i + j for i, j in pieces)
        anti_diag = set(i - j for i, j in pieces)
        
        # Calculate total squares that can be captured
        total = N * N
        captured = 0
        
        # Add rows and columns
        captured += len(rows) * N
        captured += len(cols) * N
        
        # Add diagonals and anti-diagonals
        for d in diag:
            # Count squares in this diagonal
            start_row = max(1, d - N)
            end_row = min(N, d - 1)
            captured += end_row - start_row + 1
            
        for d in anti_diag:
            # Count squares in this anti-diagonal
            start_row = max(1, d + 1)
            end_row = min(N, d + N)
            captured += end_row - start_row + 1
            
        # Remove squares that were counted multiple times
        for i, j in pieces:
            overlaps = -3  # Each piece position is counted in row, col, diag, anti-diag
            captured += overlaps
            
        result = total - captured - len(pieces)
    else:
        result = get_captured_squares(N, pieces)
    
    print(result)

if __name__ == "__main__":
    main()