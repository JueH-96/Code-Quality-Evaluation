def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    
    # We will collect all black constraints first, then check white cells last.
    # For each row r, define L[r] = the maximum column index among black cells in row r.
    # For each column c, define U[c] = the maximum row index among black cells in column c.
    # If a cell (r,c) must be white, then we cannot have BOTH L[r] >= c AND U[c] >= r.
    
    # Use dictionaries because N can be up to 10^9, and we only have up to M relevant entries.
    L = {}
    U = {}
    
    black_cells = []
    white_cells = []
    
    idx = 2
    for _ in range(M):
        x = int(input_data[idx]);   idx += 1
        y = int(input_data[idx]);   idx += 1
        color = input_data[idx];    idx += 1
        
        if color == 'B':
            black_cells.append((x, y))
        else:  # color == 'W'
            white_cells.append((x, y))
    
    # Process black cells to build L and U
    for (r, c) in black_cells:
        if r not in L:
            L[r] = c
        else:
            L[r] = max(L[r], c)
        
        if c not in U:
            U[c] = r
        else:
            U[c] = max(U[c], r)
    
    # Now check white cells
    # For each white cell (r,c), we cannot have L[r] >= c AND U[c] >= r.
    # If this happens, answer is No.
    for (r, c) in white_cells:
        # retrieve Lr, Uc (default to 0 if not in dictionary)
        Lr = L.get(r, 0)
        Uc = U.get(c, 0)
        
        if Lr >= c and Uc >= r:
            print("No")
            return
    
    # If no conflict found, print Yes
    print("Yes")

# Call the main function
main()