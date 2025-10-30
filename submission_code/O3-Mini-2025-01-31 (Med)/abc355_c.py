def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    T = int(next(it))
    
    # initialize counters
    rows = [0] * N
    cols = [0] * N
    diag1 = 0
    diag2 = 0
    
    # Process each turn
    for turn in range(1, T + 1):
        num = int(next(it))
        # Calculate the corresponding row and column (0-indexed)
        r = (num - 1) // N
        c = (num - 1) % N
        
        # Mark the cell by updating counters
        rows[r] += 1
        cols[c] += 1
        
        if r == c:
            diag1 += 1
        if r == N - 1 - c:
            diag2 += 1
        
        # Check if any condition for Bingo is satisfied
        if rows[r] == N or cols[c] == N or diag1 == N or diag2 == N:
            print(turn)
            return
    
    # If no Bingo after T turns
    print(-1)

if __name__ == '__main__':
    main()