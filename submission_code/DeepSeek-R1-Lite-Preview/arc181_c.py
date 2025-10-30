def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    P = list(map(int, N_and_rest[1:N+1]))
    Q = list(map(int, N_and_rest[N+1:2*N+1]))
    
    # Assign ranks based on P and Q
    rank_P = {p: i for i, p in enumerate(P)}
    rank_Q = {q: i for i, q in enumerate(Q)}
    
    # Assign binary strings to rows and columns based on their ranks
    row_bin = {}
    for i in range(1, N+1):
        row_bin[i] = format(rank_P[i], '0' + str(N) + 'b')
    
    col_bin = {}
    for j in range(1, N+1):
        col_bin[j] = format(rank_Q[j], '0' + str(N) + 'b')
    
    # Initialize grid with zeros
    grid = [[0 for _ in range(N)] for _ in range(N)]
    
    # Set grid cells to be consistent with row and column binary strings
    for i in range(1, N+1):
        for j in range(1, N+1):
            grid[i-1][j-1] = int(row_bin[i][j-1]) & int(col_bin[j][i-1])
    
    # Print the grid
    for row in grid:
        print(''.join(map(str, row)))

if __name__ == "__main__":
    main()