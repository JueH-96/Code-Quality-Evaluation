def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    pieces = [(int(data[i*2+2]), int(data[i*2+3])) for i in range(M)]
    
    # Sets to track which rows, columns, diagonals, and anti-diagonals are attacked
    attacked_rows = set()
    attacked_columns = set()
    attacked_diagonals = set()  # i+j
    attacked_antidiagonals = set()  # i-j
    
    for a, b in pieces:
        attacked_rows.add(a)
        attacked_columns.add(b)
        attacked_diagonals.add(a + b)
        attacked_antidiagonals.add(a - b)
    
    # Total squares in the grid
    total_squares = N * N
    
    # Calculate the number of attacked squares
    # Rows and columns are easy to count
    attacked_squares = len(attacked_rows) * N + len(attacked_columns) * N
    
    # We need to subtract the intersections of rows and columns (pieces themselves)
    attacked_squares -= len(pieces)
    
    # Calculate the number of squares attacked by diagonals and anti-diagonals
    # For each unique diagonal and anti-diagonal value, calculate how many squares are on that line
    def count_diagonal_squares(n, line):
        if 2 <= line <= 2 * n:
            if line <= n + 1:
                return line - 1
            else:
                return 2 * n - line + 1
        return 0
    
    diagonal_attacks = sum(count_diagonal_squares(N, d) for d in attacked_diagonals)
    antidiagonal_attacks = sum(count_diagonal_squares(N, d + N) for d in attacked_antidiagonals)
    
    # Add diagonal and anti-diagonal attacks to the total attacked squares
    attacked_squares += diagonal_attacks + antidiagonal_attacks
    
    # Calculate the number of safe squares
    safe_squares = total_squares - attacked_squares
    
    # Output the result
    print(safe_squares)

if __name__ == "__main__":
    main()