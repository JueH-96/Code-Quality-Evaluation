def solve():
    N = int(input())
    P = list(map(int, input().split()))
    Q = list(map(int, input().split()))
    
    # Convert to 0-based indexing
    P = [x-1 for x in P]
    Q = [x-1 for x in Q]
    
    # Create grid filled with zeros
    grid = [[0]*N for _ in range(N)]
    
    # For each row i, set grid[P[i]][j] = 1 for some columns j
    # to ensure rows are in lexicographical order
    for i in range(N-1):
        # Find first position where rows differ
        pos = 0
        while pos < N and grid[P[i]][pos] == grid[P[i+1]][pos]:
            pos += 1
            
        if pos < N:
            grid[P[i+1]][pos] = 1
    
    # For each column i, set grid[j][Q[i]] = 1 for some rows j
    # to ensure columns are in lexicographical order  
    for i in range(N-1):
        # Find first position where columns differ
        pos = 0
        while pos < N and grid[pos][Q[i]] == grid[pos][Q[i+1]]:
            pos += 1
            
        if pos < N:
            grid[pos][Q[i+1]] = 1

    # Print result
    for row in grid:
        print(''.join(map(str, row)))

solve()