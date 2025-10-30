def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    # Parse input
    it = iter(input_data)
    N = int(next(it))
    T = int(next(it))
    
    # Precompute position mapping for each number in grid.
    # Grid numbers: for cell at row i (1-indexed), col j (1-indexed)
    # number = N*(i - 1) + j
    # So, for each number from 1 to N^2, we can compute its position.
    # But since T might be less than N^2, we can also compute when announced.
    # Easier: Precompute mapping for all numbers.
    pos = [None] * (N * N + 1)
    for i in range(1, N + 1):
        base = N * (i - 1)
        for j in range(1, N + 1):
            num = base + j
            pos[num] = (i - 1, j - 1)
    
    row_count = [0] * N
    col_count = [0] * N
    diag_main = 0
    diag_anti = 0
    
    # Process each turn and update counts.
    for turn in range(1, T + 1):
        a = int(next(it))
        r, c = pos[a]  # Get the position of the number.
        # Mark the cell.
        row_count[r] += 1
        col_count[c] += 1
        if r == c:
            diag_main += 1
        if r + c == N - 1:
            diag_anti += 1
        
        # Check if any condition (row, col or one of the diagonals) is fully marked.
        if row_count[r] == N or col_count[c] == N or diag_main == N or diag_anti == N:
            sys.stdout.write(str(turn))
            return
    
    sys.stdout.write(str(-1))

if __name__ == '__main__':
    main()