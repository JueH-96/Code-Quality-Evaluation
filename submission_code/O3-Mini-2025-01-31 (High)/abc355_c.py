def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    T = int(data[1])
    moves = list(map(int, data[2:]))
    
    # Initialize counters for each row, each column, and the two diagonals
    row_counts = [0] * N
    col_counts = [0] * N
    diag_main = 0  # For the main diagonal (top-left to bottom-right)
    diag_anti = 0  # For the anti-diagonal (top-right to bottom-left)
    
    for turn, num in enumerate(moves, start=1):
        # Compute the 0-indexed row and column positions based on the problem's formula:
        # For a given number num, row = (num - 1) // N and col = (num - 1) % N.
        r = (num - 1) // N
        c = (num - 1) % N
        
        # Mark the cell by updating the corresponding row and column counters.
        row_counts[r] += 1
        if row_counts[r] == N:
            print(turn)
            return
        
        col_counts[c] += 1
        if col_counts[c] == N:
            print(turn)
            return
        
        # Update the diagonal counters if applicable.
        if r == c:
            diag_main += 1
            if diag_main == N:
                print(turn)
                return
        if r + c == N - 1:
            diag_anti += 1
            if diag_anti == N:
                print(turn)
                return
    
    # If no row, column, or diagonal gets marked completely in T turns.
    print(-1)

if __name__ == '__main__':
    main()