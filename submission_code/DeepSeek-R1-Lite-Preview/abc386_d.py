def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    black_rows = {}
    black_cols = {}
    white_cells = []
    
    index = 2
    for _ in range(M):
        X = int(data[index])
        Y = int(data[index + 1])
        C = data[index + 2]
        index += 3
        if C == 'B':
            if X not in black_rows:
                black_rows[X] = Y
            else:
                black_rows[X] = max(black_rows[X], Y)
            if Y not in black_cols:
                black_cols[Y] = X
            else:
                black_cols[Y] = max(black_cols[Y], X)
        else:
            white_cells.append((X, Y))
    
    # For rows without black cells, K_r = 0
    # For columns without black cells, L_c = 0
    # Determine K_r and L_c
    K = {}
    for r in range(1, N+1):
        K[r] = black_rows.get(r, 0)
    
    L = {}
    for c in range(1, N+1):
        L[c] = black_cols.get(c, 0)
    
    # Check each white cell
    for r, c in white_cells:
        if K[r] >= c and L[c] >= r:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()