def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Read N and Q
    N = int(input_data[0])
    Q = int(input_data[1])
    
    # Read the pattern P line by line
    lines = input_data[2 : 2 + N]
    
    # Convert pattern to a 2D array of 0/1 (white/black)
    black = [[1 if c == 'B' else 0 for c in row] for row in lines]
    
    # Build prefix sum for the NxN pattern
    # PS[i][j] = sum of black squares in subrectangle (0,0) to (i,j)
    PS = [[0]*N for _ in range(N)]
    PS[0][0] = black[0][0]
    # first row
    for j in range(1, N):
        PS[0][j] = PS[0][j-1] + black[0][j]
    
    # first column
    for i in range(1, N):
        PS[i][0] = PS[i-1][0] + black[i][0]
    
    # rest
    for i in range(1, N):
        for j in range(1, N):
            PS[i][j] = PS[i-1][j] + PS[i][j-1] - PS[i-1][j-1] + black[i][j]
    
    # sumAll = total black squares in one NxN block
    sumAll = PS[N-1][N-1]
    
    # Define a helper function f(r, c) that gives the number of black squares
    # in the rectangle from (0,0) to (r,c) inclusive in the infinite grid
    def f(r, c):
        # If out of range, no squares
        if r < 0 or c < 0:
            return 0
        
        # Number of complete blocks in row dimension
        rowBlocks = (r + 1) // N
        # Number of complete blocks in column dimension
        colBlocks = (c + 1) // N
        
        # Remainder for row and column
        rowRem = (r + 1) % N
        colRem = (c + 1) % N
        
        # The number of black squares from the complete blocks
        total = rowBlocks * colBlocks * sumAll
        
        # Add the partial block from prefix sum if remainder > 0
        if rowRem > 0 and colRem > 0:
            total += PS[rowRem - 1][colRem - 1]
        
        return total
    
    # Process queries
    # Queries start at index 2 + N in input_data
    idx = 2 + N
    out = []
    for _ in range(Q):
        A = int(input_data[idx]); B = int(input_data[idx+1])
        C = int(input_data[idx+2]); D = int(input_data[idx+3])
        idx += 4
        
        # Use the inclusion-exclusion principle with f
        ans = f(C, D) - f(A - 1, D) - f(C, B - 1) + f(A - 1, B - 1)
        out.append(str(ans))
    
    # Print answers
    print("
".join(out))


# Call solve()
if __name__ == "__main__":
    solve()